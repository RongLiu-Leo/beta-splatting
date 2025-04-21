// TODO: Fix all "gaussian" related names to "beta"

import { RadixSort } from '@/radix-sort/radixSortApple'
import {
  StaticArray,
  Struct,
  f32,
  i32,
  u32,
  vec2,
  vec3,
} from '@/splatting-web/packing'
import { PackedGaussians } from '@/splatting-web/ply'
import { RadixSortKernel } from 'webgpu-radix-sort'

import { GpuContext } from '@/beta-preprocessor/gpuContext'
import { Renderer } from '@/splatting-web/renderer'

import betaShader from './beta.wgsl?raw'
import intersectionOffsetPrefixSumShader from './betaIntersectionOffsetPrefixSum.wgsl?raw'
import prefixSumShader from './betaPrefixSum.wgsl?raw'
import tileDepthShader from './betaTileDepthKey.wgsl?raw'

function nextPowerOfTwo(x: number): number {
  return Math.pow(2, Math.ceil(Math.log2(x)))
}

const resultLayout = new Struct([
  ['id', i32],
  ['radii', i32],
  ['depth', f32],
  ['tiles_touched', u32],
  ['cum_tiles_touched', u32],
  ['uv', new vec2(f32)],
  ['conic', new vec3(f32)],
  ['color', new vec3(f32)],
  ['opacity', f32],
  ['beta', f32],
])

const auxLayout = new Struct([
  ['num_intersections', u32],
  ['num_visible_betas', u32],
  ['min_depth', i32],
  ['max_depth', i32],
])

const prefixLayout = u32

export const tileDepthKeyLayout = new Struct([
  ['key', u32],
  ['beta_id', u32],
])

// const numTilesArray = nextPowerOfTwo(1 * 1e6)
// const numTilesArray = 3 * 1e6
// const numTilesArray = nextPowerOfTwo(3 * 1e6)
// const numTilesArray = 6 * 1e6
// const numTilesArray = nextPowerOfTwo(6 * 1e6)
const numTilesArray = 14 * 1e6
// const numTilesArray = 16 * 1e6

export const tileDepthKeyArrayLayoutTest = new StaticArray(
  tileDepthKeyLayout,
  numTilesArray,
)

export class Preprocessor {
  context: GpuContext

  gaussians: PackedGaussians
  numBetas: number
  pointDataBuffer: GPUBuffer
  numPadded: number

  numThreads: number

  pipelineLayout: GPUPipelineLayout
  pipeline: GPUComputePipeline
  prefixSumPipeline: GPUComputePipeline
  prefixSumSyncPipeline: GPUComputePipeline
  tileDepthKeyPipeline: GPUComputePipeline
  intersectionOffsetPrefixSumPipeline: GPUComputePipeline
  // @ts-ignore
  bindGroup: GPUBindGroup

  resultBuffer: GPUBuffer
  uniformBuffer: GPUBuffer
  tileDepthKeyBuffer: GPUBuffer

  auxBuffer: GPUBuffer
  auxBufferRead: GPUBuffer
  intersectionOffsetBuffer: GPUBuffer
  emptyPrefixBuffer: GPUBuffer
  prefixBuffer: GPUBuffer

  public resultArrayLayout: StaticArray
  numIntersections = 0
  public tileDepthKeyArrayLayout: StaticArray
  public prefixArrayLayout: StaticArray
  public intersectionOffsetArrayLayout: StaticArray
  public cumSumTiles: number[] = []

  maxBufferSize: number
  numTiles: number

  radixSorter: RadixSort
  radixSortKernel: typeof RadixSortKernel

  renderer: Renderer

  constructor(
    context: GpuContext,
    gaussians: PackedGaussians,
    pointDataBuffer: GPUBuffer,
    uniformBuffer: GPUBuffer,
    canvas: HTMLCanvasElement,
    renderer: Renderer,
  ) {
    this.context = context
    console.log('Device limits:', this.context.device.limits)

    this.maxBufferSize = this.context.device.limits.maxBufferSize

    this.renderer = renderer

    this.gaussians = gaussians
    this.numBetas = gaussians.numGaussians
    this.numPadded = nextPowerOfTwo(gaussians.numGaussians)

    console.log('this.numBetas', this.numBetas)

    this.pointDataBuffer = pointDataBuffer
    this.uniformBuffer = uniformBuffer

    this.numThreads = 2048

    const numtilesX = Math.ceil(canvas.width / 16)
    const numtilesY = Math.ceil(canvas.height / 16)
    this.numTiles = numtilesX * numtilesY

    const itemsPerThread = Math.ceil(this.numBetas / this.numThreads)

    this.resultArrayLayout = new StaticArray(resultLayout, this.numBetas)
    this.tileDepthKeyArrayLayout = new StaticArray(
      tileDepthKeyLayout,
      numTilesArray,
    )

    this.prefixArrayLayout = new StaticArray(prefixLayout, this.numThreads)

    this.intersectionOffsetArrayLayout = new StaticArray(u32, this.numTiles)

    this.resultBuffer = this.context.device.createBuffer({
      size: this.resultArrayLayout.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
      label: 'preprocessor.resultBuffer',
    })

    this.auxBuffer = this.context.device.createBuffer({
      size: auxLayout.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
      label: 'preprocessor.auxBuffer',
    })

    this.auxBufferRead = this.context.device.createBuffer({
      size: auxLayout.size,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      mappedAtCreation: false,
      label: 'preprocessor.auxBufferRead',
    })

    this.intersectionOffsetBuffer = this.context.device.createBuffer({
      size: this.intersectionOffsetArrayLayout.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
      label: 'preprocessor.intersectionOffsetBuffer',
    })

    this.emptyPrefixBuffer = this.context.device.createBuffer({
      size: this.prefixArrayLayout.size,
      usage: GPUBufferUsage.COPY_SRC,
      mappedAtCreation: false,
    })

    this.prefixBuffer = this.context.device.createBuffer({
      size: this.prefixArrayLayout.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
      label: 'preprocessor.prefixBuffer',
    })

    this.tileDepthKeyBuffer = this.context.device.createBuffer({
      size: this.tileDepthKeyArrayLayout.size,
      usage:
        GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
      mappedAtCreation: false,
      label: 'preprocessor.tileDepthKeyBuffer',
    })

    this.radixSorter = new RadixSort(
      context,
      this.tileDepthKeyArrayLayout.nElements,
      2,
      this.renderer,
    )

    this.radixSortKernel = new RadixSortKernel({
      device: this.context.device, // GPUDevice to use
      keys: this.tileDepthKeyBuffer, // GPUBuffer containing the keys to sort
      // keys: keysBuffer, // GPUBuffer containing the keys to sort
      // values: valuesBuffer, // (optional) GPUBuffer containing the associated values
      count: this.tileDepthKeyArrayLayout.nElements, // Number of elements to sort
      // count: keys.length, // Number of elements to sort
      check_order: false, // Whether to check if the input is already sorted to exit early
      bit_count: 32, // Number of bits per element. Must be a multiple of 4 (default: 32)
      workgroup_size: { x: 16, y: 16 }, // Workgroup size in x and y dimensions. (x * y) must be a power of two
    })

    console.log('this.radixSortKernel', this.radixSortKernel)

    const computeBindGroupLayout = this.createUniforms()

    this.pipelineLayout = this.context.device.createPipelineLayout({
      bindGroupLayouts: [computeBindGroupLayout],
    })

    const gaussShaderWithParams = gaussShader
      .replace(
        'const item_per_thread: i32 = 1;',
        `const item_per_thread: i32 = ${itemsPerThread};`,
      )
      .replace(
        'const num_quads_unpaddded: i32 = 1;',
        `const num_quads_unpaddded: i32 = ${this.numBetas};`,
      )

    this.pipeline = this.context.device.createComputePipeline({
      layout: this.pipelineLayout,
      compute: {
        module: this.context.device.createShaderModule({
          code: gaussShaderWithParams,
        }),
        entryPoint: 'main',
        constants: {
          // num_quads_unpaddded: this.numBetas,
        },
      },
    })

    const intersectionOffsetPrefixSumShaderWithParams =
      intersectionOffsetPrefixSumShader.replace(
        'const num_tiles: i32 = 1;',
        `const num_tiles: i32 = ${this.numTiles};`,
      )

    this.intersectionOffsetPrefixSumPipeline =
      this.context.device.createComputePipeline({
        layout: this.pipelineLayout,
        compute: {
          module: this.context.device.createShaderModule({
            code: intersectionOffsetPrefixSumShaderWithParams,
          }),
          entryPoint: 'main',
          constants: {
            // num_quads_unpaddded: this.numBetas,
          },
        },
      })

    const prefixSumShaderWithParams = prefixSumShader
      .replace(
        'const item_per_thread: i32 = 1;',
        `const item_per_thread: i32 = ${itemsPerThread};`,
      )
      .replace(
        'const num_quads_unpaddded: i32 = 1;',
        `const num_quads_unpaddded: i32 = ${this.numBetas};`,
      )

    this.prefixSumPipeline = this.context.device.createComputePipeline({
      layout: this.pipelineLayout,
      compute: {
        module: this.context.device.createShaderModule({
          code: prefixSumShaderWithParams,
        }),
        entryPoint: 'main',
      },
    })

    this.prefixSumSyncPipeline = this.context.device.createComputePipeline({
      layout: this.pipelineLayout,
      compute: {
        module: this.context.device.createShaderModule({
          code: prefixSumShaderWithParams,
        }),
        entryPoint: 'prefix_sum_sync',
      },
    })

    let newTileDepthShader = tileDepthShader.replace(
      'const item_per_thread: i32 = 1;',
      `const item_per_thread: i32 = ${itemsPerThread};`,
    )
    newTileDepthShader = newTileDepthShader.replace(
      'const num_quads_unpaddded: i32 = 1;',
      `const num_quads_unpaddded: i32 = ${this.numBetas};`,
    )

    this.tileDepthKeyPipeline = this.context.device.createComputePipeline({
      layout: this.pipelineLayout,
      compute: {
        module: this.context.device.createShaderModule({
          code: newTileDepthShader,
        }),
        entryPoint: 'main',
      },
    })

    console.info('Preprocessor initialized')
  }

  private createUniforms() {
    const computeBindGroupLayout = this.context.device.createBindGroupLayout({
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'read-only-storage',
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
            // hasDynamicOffset: true,
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
        {
          binding: 6,
          visibility: GPUShaderStage.COMPUTE,
          buffer: {
            type: 'storage',
          },
        },
      ],
    })

    this.bindGroup = this.context.device.createBindGroup({
      layout: computeBindGroupLayout,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.pointDataBuffer,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.uniformBuffer,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.resultBuffer,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.tileDepthKeyBuffer,
          },
        },
        {
          binding: 4,
          resource: {
            buffer: this.prefixBuffer,
          },
        },
        {
          binding: 5,
          resource: {
            buffer: this.auxBuffer,
          },
        },
        {
          binding: 6,
          resource: {
            buffer: this.intersectionOffsetBuffer,
          },
        },
      ],
    })

    console.log('this.resultArrayLayout', this.resultArrayLayout)

    console.log('Uniforms initialized')

    return computeBindGroupLayout
  }

  public destroy() {}

  async run() {
    const commandEncoder = this.context.device.createCommandEncoder()

    this.renderer.timestamp(commandEncoder, 'preprocess start')

    commandEncoder.clearBuffer(this.resultBuffer)
    commandEncoder.clearBuffer(this.tileDepthKeyBuffer)
    commandEncoder.clearBuffer(this.prefixBuffer)
    commandEncoder.clearBuffer(this.auxBuffer)
    commandEncoder.clearBuffer(this.intersectionOffsetBuffer)

    this.renderer.timestamp(commandEncoder, 'clear')

    const betaEncoder = commandEncoder.beginComputePass()
    betaEncoder.setPipeline(this.pipeline)
    betaEncoder.setBindGroup(0, this.bindGroup)
    betaEncoder.dispatchWorkgroups(Math.ceil(this.numBetas / 256) + 1)
    betaEncoder.end()

    this.renderer.timestamp(commandEncoder, 'beta properties')

    const intersectionOffsetPrefixSumEncoder = commandEncoder.beginComputePass()
    intersectionOffsetPrefixSumEncoder.setPipeline(
      this.intersectionOffsetPrefixSumPipeline,
    )
    intersectionOffsetPrefixSumEncoder.setBindGroup(0, this.bindGroup)
    intersectionOffsetPrefixSumEncoder.dispatchWorkgroups(1)
    intersectionOffsetPrefixSumEncoder.end()

    this.renderer.timestamp(commandEncoder, 'intersection offset prefix sum')

    const prefixSumPass = commandEncoder.beginComputePass()
    prefixSumPass.setPipeline(this.prefixSumPipeline)
    prefixSumPass.setBindGroup(0, this.bindGroup)
    prefixSumPass.dispatchWorkgroups(Math.ceil(this.numBetas / 256) + 1)
    prefixSumPass.end()

    this.renderer.timestamp(commandEncoder, 'prefix sum scan')

    const prefixSumSyncPass = commandEncoder.beginComputePass()
    prefixSumSyncPass.setPipeline(this.prefixSumSyncPipeline)
    prefixSumSyncPass.setBindGroup(0, this.bindGroup)
    prefixSumSyncPass.dispatchWorkgroups(this.numThreads)
    prefixSumSyncPass.end()

    this.renderer.timestamp(commandEncoder, 'prefix sum sync')

    const tileDepthKeyPassEncoder = commandEncoder.beginComputePass()
    tileDepthKeyPassEncoder.setPipeline(this.tileDepthKeyPipeline)
    tileDepthKeyPassEncoder.setBindGroup(0, this.bindGroup)
    tileDepthKeyPassEncoder.dispatchWorkgroups(this.numBetas / 256 + 1)
    tileDepthKeyPassEncoder.end()

    this.renderer.timestamp(commandEncoder, 'tile depth key')

    commandEncoder.copyBufferToBuffer(
      this.auxBuffer,
      0,
      this.auxBufferRead,
      0,
      auxLayout.size,
    )

    this.context.device.queue.submit([commandEncoder.finish()])

    await this.auxBufferRead.mapAsync(GPUMapMode.READ, 0, 4)
    this.numIntersections = new Uint32Array(
      this.auxBufferRead.getMappedRange(0, 4),
    )[0]!

    console.log('numIntersections', this.numIntersections)

    const { values: sortedValuesRadix } = this.radixSorter.sort(
      this.tileDepthKeyBuffer,
      numTilesArray,
      // this.numIntersections,
    )

    ///
    // const encoder = this.context.device.createCommandEncoder()
    // const pass = encoder.beginComputePass()
    // this.radixSortKernel.dispatch(pass) // Sort keysBuffer and valuesBuffer in-place on the GPU
    // pass.end()
    ///

    this.auxBufferRead.unmap()

    return {
      betaData: this.resultBuffer,
      betaDataLayout: this.resultArrayLayout,
      intersectionKeys: sortedValuesRadix,
      intersectionKeysLayout: this.tileDepthKeyArrayLayout,
      intersectionOffsets: this.intersectionOffsetBuffer,
      intersectionOffsetsLayout: this.intersectionOffsetArrayLayout,
      aux: this.auxBuffer,
      auxLayout,
    }
  }
}
