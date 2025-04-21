// This file contains the main rendering code. Unlike the official implementation,
// instead of using compute shaders and iterating through (possibly) all gaussians,
// we instead use a vertex shader to turn each gaussian into a quad facing the camera
// and then use the fragment shader to paint the gaussian on the quad.
// If we draw the quads in order of depth, with well chosen blending settings we can
// get the same color accumulation rule as in the original paper.
// This approach is faster than the original implementation on webGPU but still substantially
// slow compared to the CUDA impl. The main bottleneck is the sorting of the quads by depth,
// which is done on the CPU but could presumably be replaced by a compute shader sort.

import { GpuContext } from '@/point-preprocessor/gpuContext'
import { Mat4 } from 'wgpu-matrix'
import { InteractiveCamera } from './camera'
import { Mat4x4, Struct, f32, u32, vec2, vec3 } from './packing'
import { PackedGaussians } from './ply'
import computeKernel from './shaders/rasterize.wgsl?raw'
import screenShader from './shaders/screen.wgsl?raw'

import { Preprocessor } from '@/point-preprocessor/preproces'

const uniformLayout = new Struct([
  ['viewMatrix', new Mat4x4(f32)],
  ['projMatrix', new Mat4x4(f32)],
  ['cameraPosition', new vec3(f32)],
  ['tanHalfFovX', f32],
  ['tanHalfFovY', f32],
  ['focalX', f32],
  ['focalY', f32],
  ['scaleModifier', f32],
  ['screen_size', new vec2(u32)],
])

function mat4toArrayOfArrays(m: Mat4): number[][] {
  return [
    [m[0]!, m[1]!, m[2]!, m[3]!],
    [m[4]!, m[5]!, m[6]!, m[7]!],
    [m[8]!, m[9]!, m[10]!, m[11]!],
    [m[12]!, m[13]!, m[14]!, m[15]!],
  ]
}

export class Renderer {
  canvas: HTMLCanvasElement
  interactiveCamera: InteractiveCamera
  numGaussians: number

  context: GpuContext
  contextGpu: GPUCanvasContext
  presentationFormat: GPUTextureFormat = 'rgba16float'

  uniformBuffer: GPUBuffer
  pointDataBuffer: GPUBuffer

  colorBuffer: GPUTexture
  colorBufferView: GPUTextureView
  sampler: GPUSampler

  computeBindGroup: GPUBindGroup
  computeScreenBindGroup: GPUBindGroup
  computeBindGroupLayout: GPUBindGroupLayout

  computePipeline: GPUComputePipeline
  computeRenderPipeline: GPURenderPipeline

  depthSortMatrix: number[][] = []

  fpsCounter: HTMLLabelElement
  lastDraw: number

  destroyCallback: (() => void) | null = null

  frameTime: number

  preprocessor: Preprocessor

  numTimestamps = 40
  currentTimeStamp = 0
  timeStampQuerySet: GPUQuerySet | null = null
  timeStampBuffer: GPUBuffer
  timeStampLabels: string[] = []

  invalid = false
  isAnimating = false

  public static async requestContext(gaussians: PackedGaussians) {
    const gpu = navigator.gpu
    if (!gpu) {
      throw new Error(
        'WebGPU not supported on this browser! (navigator.gpu is null)',
      )
    }

    const adapter = await gpu.requestAdapter()

    if (!adapter) {
      throw new Error(
        'WebGPU not supported on this browser! (gpu.adapter is null)',
      )
    }

    // for good measure, we request 1.5 times the amount of memory we need
    const byteLength = gaussians.gaussiansBuffer.byteLength
    let device
    const common = {
      requiredLimits: {
        maxStorageBufferBindingSize: 1.5 * byteLength,
        maxBufferSize: 1.5 * byteLength,
      },
    }

    try {
      device = await adapter.requestDevice({
        ...common,
        requiredFeatures: ['timestamp-query'],
      })
    } catch (error) {
      device = await adapter.requestDevice({
        ...common,
      })
    }

    return new GpuContext(gpu, adapter, device)
  }

  // destroy the renderer and return a promise that resolves when it's done (after the next frame)
  public async destroy(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.destroyCallback = resolve
    })
  }

  constructor(
    canvas: HTMLCanvasElement,
    interactiveCamera: InteractiveCamera,
    gaussians: PackedGaussians,
    context: GpuContext,
    fpsCounter: HTMLLabelElement,
  ) {
    console.log('gaussians', gaussians)

    this.canvas = canvas
    this.interactiveCamera = interactiveCamera
    this.context = context
    const contextGpu = canvas.getContext('webgpu')
    if (!contextGpu) {
      throw new Error('WebGPU context not found!')
    }
    this.contextGpu = contextGpu
    this.fpsCounter = fpsCounter
    this.lastDraw = performance.now()

    this.numGaussians = gaussians.numGaussians

    this.frameTime = 0
    if (this.context.hasTimestampQuery) {
      this.timeStampQuerySet = this.context.device.createQuerySet({
        type: 'timestamp',
        count: this.numTimestamps,
      })
    }

    this.timeStampBuffer = this.context.device.createBuffer({
      size: 8 * this.numTimestamps,
      usage:
        GPUBufferUsage.QUERY_RESOLVE |
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    })

    this.contextGpu.configure({
      device: this.context.device,
      format: this.presentationFormat,
      alphaMode: 'premultiplied',
    })

    this.pointDataBuffer = this.context.device.createBuffer({
      size: gaussians.gaussianArrayLayout.size,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
      label: 'renderer.pointDataBuffer',
    })
    new Uint8Array(this.pointDataBuffer.getMappedRange()).set(
      new Uint8Array(gaussians.gaussiansBuffer),
    )
    this.pointDataBuffer.unmap()

    // Create a GPU buffer for the uniform data.
    this.uniformBuffer = this.context.device.createBuffer({
      size: uniformLayout.size,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
      label: 'renderer.uniformBuffer',
    })

    this.colorBuffer = this.context.device.createTexture({
      size: {
        width: canvas.width,
        height: canvas.height,
      },
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.STORAGE_BINDING |
        GPUTextureUsage.TEXTURE_BINDING,
    })

    this.colorBufferView = this.colorBuffer.createView()

    this.computeBindGroupLayout = this.context.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: {
            access: 'write-only',
            format: 'rgba8unorm',
            viewDimension: '2d',
          },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'uniform',
          },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
          },
        },
        {
          binding: 3,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
          },
        },
        {
          binding: 4,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
          },
        },
        {
          binding: 5,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
          },
        },
      ],
    })

    this.preprocessor = new Preprocessor(
      this.context,
      gaussians,
      this.pointDataBuffer,
      this.uniformBuffer,
      this.canvas,
      this,
    )

    this.computeBindGroup = this.context.device.createBindGroup({
      layout: this.computeBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: this.colorBufferView,
        },
        {
          binding: 1,
          resource: { buffer: this.uniformBuffer },
        },
        {
          binding: 2,
          resource: { buffer: this.preprocessor.resultBuffer },
        },
        {
          binding: 3,
          resource: { buffer: this.preprocessor.tileDepthKeyBuffer },
        },
        {
          binding: 4,
          resource: { buffer: this.preprocessor.intersectionOffsetBuffer },
        },
        {
          binding: 5,
          resource: { buffer: this.preprocessor.auxBuffer },
        },
      ],
    })

    console.log('this.computeBindGroup', this.computeBindGroup)

    // COMPUTE PIPELINE

    const computePipelineLayout = this.context.device.createPipelineLayout({
      bindGroupLayouts: [this.computeBindGroupLayout],
    })

    this.computePipeline = this.context.device.createComputePipeline({
      layout: computePipelineLayout,
      compute: {
        module: this.context.device.createShaderModule({
          code: computeKernel.replace(
            'const intersection_array_length = 1;',
            `const intersection_array_length = ${
              this.preprocessor.tileDepthKeyBuffer.size / (4 * 2)
            };`,
          ),
        }),
        entryPoint: 'main',
      },
      label: 'computePipeline',
    })

    // COMPUTE SCREEN

    this.sampler = this.context.device.createSampler({
      addressModeU: 'repeat',
      addressModeV: 'repeat',
      magFilter: 'linear',
      minFilter: 'nearest',
      mipmapFilter: 'nearest',
      maxAnisotropy: 1,
    })

    const computeScreenBindGroupLayout =
      this.context.device.createBindGroupLayout({
        entries: [
          {
            binding: 0,
            visibility: GPUShaderStage.FRAGMENT,
            sampler: {
              // type: 'non-filtering',
            },
          },
          {
            binding: 1,
            visibility: GPUShaderStage.FRAGMENT,
            texture: {},
          },
        ],
      })

    this.computeScreenBindGroup = this.context.device.createBindGroup({
      layout: computeScreenBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: this.sampler,
        },
        {
          binding: 1,
          resource: this.colorBufferView,
        },
      ],
    })

    const computeScreenPipelineLayout =
      this.context.device.createPipelineLayout({
        bindGroupLayouts: [computeScreenBindGroupLayout],
      })

    this.computeRenderPipeline = this.context.device.createRenderPipeline({
      layout: computeScreenPipelineLayout,
      vertex: {
        module: this.context.device.createShaderModule({
          code: screenShader,
        }),
        entryPoint: 'vert_main',
      },
      fragment: {
        module: this.context.device.createShaderModule({
          code: screenShader,
        }),
        entryPoint: 'frag_main',
        targets: [
          {
            format: this.presentationFormat,
          },
        ],
      },
      primitive: {
        topology: 'triangle-list',
      },
    })

    // start the animation loop
    requestAnimationFrame(() => this.animate(true))
  }

  timestamp(encoder: GPUCommandEncoder, label: string) {
    // if (!this.timeStampQuerySet) return
    // encoder.writeTimestamp(this.timeStampQuerySet, this.currentTimeStamp)
    // this.timeStampLabels[this.currentTimeStamp] = label
    // this.currentTimeStamp++
    // if (this.currentTimeStamp >= this.numTimestamps) {
    //   this.currentTimeStamp = 0
    // }
  }

  async readBuffer(device: GPUDevice, buffer: GPUBuffer) {
    const size = buffer.size
    // Get a GPU buffer for reading in an unmapped state.
    const gpuReadBuffer = device.createBuffer({
      size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    })

    const copyEncoder = device.createCommandEncoder()
    copyEncoder.copyBufferToBuffer(buffer, 0, gpuReadBuffer, 0, size)

    // Submit copy commands.
    const copyCommands = copyEncoder.finish()
    device.queue.submit([copyCommands])

    await gpuReadBuffer.mapAsync(GPUMapMode.READ)
    return gpuReadBuffer.getMappedRange()
  }

  printTimestampsWithLabels(timingsNanoseconds: BigInt64Array) {
    // if (!this.timeStampQuerySet) return
    // console.log('==========')
    // // Convert list of nanosecond timestamps to diffs in milliseconds
    // const timeDiffs = []
    // for (let i = 1; i < timingsNanoseconds.length; i++) {
    //   let diff = Number(timingsNanoseconds[i]! - timingsNanoseconds[i - 1]!)
    //   diff /= 1_000_000
    //   timeDiffs.push(diff)
    // }
    // // Print each diff with its associated label
    // for (let i = 0; i < timeDiffs.length; i++) {
    //   const time = timeDiffs[i]
    //   const label = this.timeStampLabels[i + 1]
    //   if (label) {
    //     console.log(label, time?.toFixed(2) + 'ms')
    //   } else {
    //     console.log(i, time?.toFixed(2) + 'ms')
    //   }
    // }
    // console.log('==========')
  }

  private destroyImpl() {
    if (this.destroyCallback === null) {
      throw new Error('destroyImpl called without destroyCallback set!')
    }

    this.uniformBuffer.destroy()
    this.pointDataBuffer.destroy()
    this.context.destroy()
    this.destroyCallback()
  }

  getFrameTime() {
    // fps counter
    const now = performance.now()
    this.frameTime = now - this.lastDraw
    const fps = 1000 / this.frameTime
    this.lastDraw = now
    this.fpsCounter.innerText =
      'FPS: ' +
      fps.toFixed(2) +
      '\nFrame time: ' +
      this.frameTime.toFixed(2) +
      'ms'
    this.fpsCounter.style.display = 'block'
  }

  async draw(nextFrameCallback: FrameRequestCallback) {
    const commandEncoder = this.context.device.createCommandEncoder()
    this.timestamp(commandEncoder, 'init')

    await this.preprocessor.run()

    const computePass = commandEncoder.beginComputePass()

    computePass.setPipeline(this.computePipeline)
    computePass.setBindGroup(0, this.computeBindGroup)
    computePass.dispatchWorkgroups(
      this.canvas.width / 16,
      this.canvas.height / 16,
      1,
    )

    computePass.end()
    this.timestamp(commandEncoder, 'render')

    const textureView = this.contextGpu.getCurrentTexture().createView()

    const renderPass = commandEncoder.beginRenderPass({
      colorAttachments: [
        {
          view: textureView,
          clearValue: { r: 0, g: 0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    })

    renderPass.setPipeline(this.computeRenderPipeline)
    renderPass.setBindGroup(0, this.computeScreenBindGroup)

    renderPass.draw(6, 1, 0, 0)
    renderPass.end()
    this.timestamp(commandEncoder, 'draw')

    if (this.timeStampQuerySet) {
      commandEncoder.resolveQuerySet(
        this.timeStampQuerySet,
        0,
        this.numTimestamps,
        this.timeStampBuffer,
        0,
      )
    }

    this.context.device.queue.submit([commandEncoder.finish()])

    this.getFrameTime()

    let printedTimestamps = false
    // Print just once for readability
    if (!printedTimestamps) {
      // Read the storage buffer data
      const arrayBuffer = await this.readBuffer(
        this.context.device,
        this.timeStampBuffer,
      )
      // Decode it into an array of timestamps in nanoseconds
      const timingsNanoseconds = new BigInt64Array(arrayBuffer)
      // Print the diff's with labels
      this.printTimestampsWithLabels(timingsNanoseconds)

      printedTimestamps = true

      this.timeStampLabels = []
      this.currentTimeStamp = 0
    }

    requestAnimationFrame(nextFrameCallback)
  }

  public invalidate() {
    this.invalid = true
  }

  async animate(forceDraw?: boolean) {
    this.isAnimating = true

    if (this.destroyCallback !== null) {
      this.destroyImpl()
      this.isAnimating = false
      return
    }

    if (!this.interactiveCamera.isDirty() && !forceDraw && !this.invalid) {
      requestAnimationFrame(() => this.animate())
      this.isAnimating = false
      return
    }

    this.invalid = false

    const camera = this.interactiveCamera.getCamera()

    const position = camera.getPosition()

    const tanHalfFovX = (0.5 * this.canvas.width) / camera.focalX
    const tanHalfFovY = (0.5 * this.canvas.height) / camera.focalY

    const uniformsMatrixBuffer = new ArrayBuffer(this.uniformBuffer.size)
    const uniforms = {
      viewMatrix: mat4toArrayOfArrays(camera.viewMatrix),
      projMatrix: mat4toArrayOfArrays(camera.getProjMatrix()),
      cameraPosition: Array.from(position),
      tanHalfFovX,
      tanHalfFovY,
      focalX: camera.focalX,
      focalY: camera.focalY,
      scaleModifier: camera.scaleModifier,
      screen_size: [this.canvas.width, this.canvas.height],
    }
    uniformLayout.pack(0, uniforms, new DataView(uniformsMatrixBuffer))

    this.context.device.queue.writeBuffer(
      this.uniformBuffer,
      0,
      uniformsMatrixBuffer,
      0,
      uniformsMatrixBuffer.byteLength,
    )

    this.draw(() => this.animate())
  }
}
