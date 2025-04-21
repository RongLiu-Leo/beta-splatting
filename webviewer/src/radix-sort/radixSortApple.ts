import { GpuContext } from '@/point-preprocessor/gpuContext'
import shaderScan from '@/radix-sort/radixAppleScan.wgsl?raw'
import shaderScanGroups from '@/radix-sort/radixAppleScanGroups.wgsl?raw'
import shaderScanSync from '@/radix-sort/radixAppleScanSync.wgsl?raw'
import { Renderer } from '@/splatting-web/renderer'

const WORKGROUP_SIZE = 256
const NUM_BITS_IN_CHUNK = 4

function get_shader_radix_scan(
  num_workgroups: number,
  numInputs: number,
  sync = false,
  groups = false,
) {
  const workgroup_size_2x = WORKGROUP_SIZE * 2
  const last_wg_padding = num_workgroups * WORKGROUP_SIZE - numInputs

  const twoPowBits = Math.pow(2, NUM_BITS_IN_CHUNK)
  const num_wg_prefixes =
    num_workgroups * Math.max(2, NUM_BITS_IN_CHUNK) * NUM_BITS_IN_CHUNK

  let shader = ''

  if (sync) {
    shader = shaderScanSync
  } else {
    shader = shaderScan
  }

  if (groups) shader = shaderScanGroups

  return shader
    .replace(
      'const workgroup_size = 1;',
      `const workgroup_size = ${WORKGROUP_SIZE};`,
    )
    .replace(
      'const workgroup_size_2x = 1;',
      `const workgroup_size_2x = ${workgroup_size_2x};`,
    )
    .replace(
      'const workgroup_cache_size = 1;',
      `const workgroup_cache_size = ${WORKGROUP_SIZE * NUM_BITS_IN_CHUNK};`,
    )
    .replace(
      'const num_bits_in_chunk = 1;',
      `const num_bits_in_chunk = ${NUM_BITS_IN_CHUNK};`,
    )
    .replace('const two_pow_bits = 16;', `const two_pow_bits = ${twoPowBits};`)
    .replace(
      'const num_workgroups_global = 1;',
      `const num_workgroups_global = ${num_workgroups};`,
    )
    .replace(
      'const num_workgroups_global_2x = 1;',
      `const num_workgroups_global_2x = ${num_workgroups * 2};`,
    )
    .replace(
      'const last_wg_padding = 1;',
      `const last_wg_padding = ${last_wg_padding};`,
    )
    .replace('const num_inputs = 1;', `const num_inputs = ${numInputs};`)
    .replace(
      'const num_wg_prefixes = 1;',
      `const num_wg_prefixes = ${num_wg_prefixes};`,
    )
    .replace(
      'const histogram_size = 512;',
      `const histogram_size = ${WORKGROUP_SIZE * twoPowBits};`,
    )
    .replaceAll(
      '@compute @workgroup_size(256, 1, 1)',
      `@compute @workgroup_size(${WORKGROUP_SIZE}, 1, 1)`,
    )
}

function GetPipelineRadixScan(
  engine_ctx: GpuContext,
  num_workgroups: number,
  numInputs: number,
) {
  if (!engine_ctx.device) throw new Error('engine_ctx.device is null')

  const shaderModule = engine_ctx.device.createShaderModule({
    code: get_shader_radix_scan(num_workgroups, numInputs),
  })
  const bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.radixScan]
  const pipelineLayoutDesc = { bindGroupLayouts }
  const layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc)

  return engine_ctx.device.createComputePipeline({
    layout,
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  })
}

function GetPipelineWGScan(
  engine_ctx: GpuContext,
  num_workgroups: number,
  numInputs: number,
) {
  if (!engine_ctx.device) throw new Error('engine_ctx.device is null')

  const shaderModule = engine_ctx.device.createShaderModule({
    code: get_shader_radix_scan(num_workgroups, numInputs, false, true),
  })
  const bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.radixScan]
  const pipelineLayoutDesc = { bindGroupLayouts }
  const layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc)

  return engine_ctx.device.createComputePipeline({
    layout,
    compute: {
      module: shaderModule,
      entryPoint: 'main',
    },
  })
}

function GetPipelineRadixScanSync(
  engine_ctx: GpuContext,
  num_workgroups: number,
  numInputs: number,
) {
  if (!engine_ctx.device) throw new Error('engine_ctx.device is null')

  const shaderModule = engine_ctx.device.createShaderModule({
    code: get_shader_radix_scan(num_workgroups, numInputs, true),
  })
  const bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.radixScan]
  const pipelineLayoutDesc = { bindGroupLayouts }
  const layout = engine_ctx.device.createPipelineLayout(pipelineLayoutDesc)

  return engine_ctx.device.createComputePipeline({
    layout,
    compute: {
      module: shaderModule,
      entryPoint: 'sync',
    },
  })
}

export class RadixSort {
  engine_ctx: GpuContext

  // buf_data: GPUBuffer
  buf_tmp: GPUBuffer[]

  bits = 32
  bitsOffset = 0
  buf_constant_radix_scan: GPUBuffer

  prefix_wg_buffer: GPUBuffer

  buf_constant_radix_scatter: GPUBuffer
  buf_download: GPUBuffer
  layout_entries_radix_scan: globalThis.GPUBindGroupLayoutEntry[]
  bindGroupLayoutRadixScan: GPUBindGroupLayout
  pipeline_radix_scan: GPUComputePipeline
  pipeline_radix_scan_sync: GPUComputePipeline

  pipeline_wg_scan: GPUComputePipeline
  count: number
  numInts: number
  num_groups_radix_scan: number
  num_groups_scatter: number
  bind_group_radix_scan: GPUBindGroup[]
  bind_group_radix_scatter: GPUBindGroup[]

  renderer: Renderer

  constructor(
    engine_ctx: GpuContext,
    numElements: number,
    numInts: number,
    renderer: Renderer,
  ) {
    this.renderer = renderer

    this.engine_ctx = engine_ctx
    this.numInts = numInts

    this.count = numElements

    this.num_groups_radix_scan = Math.floor(
      (this.count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
    )

    console.log('num elemnts with WG padding', this.num_groups_radix_scan * 256)
    console.log(
      'num padded elements',
      this.num_groups_radix_scan * 256 - this.count,
    )

    this.num_groups_scatter = Math.floor(
      (this.count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE,
    )
    console.log('this.num_groups_scatter', this.num_groups_scatter)

    // this.buf_data = engine_ctx.createBuffer0(
    //   this.count * 4 * numInts,
    //   GPUBufferUsage.STORAGE |
    //     GPUBufferUsage.COPY_SRC |
    //     GPUBufferUsage.COPY_DST,
    // )

    this.buf_tmp = new Array(2)
    this.buf_tmp[0] = engine_ctx.createBuffer0(
      this.count * 4 * numInts,
      GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    )
    this.buf_tmp[1] = engine_ctx.createBuffer0(
      this.count * 4 * numInts,
      GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_SRC |
        GPUBufferUsage.COPY_DST,
    )

    this.buf_constant_radix_scan = engine_ctx.createBuffer0(
      16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    )
    console.log('num_groups_radix_scan', this.num_groups_radix_scan)

    this.prefix_wg_buffer = engine_ctx.createBuffer0(
      this.num_groups_radix_scan *
        4 *
        Math.max(2, NUM_BITS_IN_CHUNK) *
        NUM_BITS_IN_CHUNK,
      GPUBufferUsage.STORAGE |
        GPUBufferUsage.COPY_DST |
        GPUBufferUsage.COPY_SRC,
    )

    this.buf_constant_radix_scatter = engine_ctx.createBuffer0(
      16,
      GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    )

    this.buf_download = engine_ctx.createBuffer0(
      this.count * 4 * numInts,
      GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    )

    this.layout_entries_radix_scan = [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'uniform',
        },
      },
      {
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: 'read-only-storage',
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
    ]

    if (!engine_ctx.device) throw new Error('engine_ctx.device is null')

    this.bindGroupLayoutRadixScan = engine_ctx.device.createBindGroupLayout({
      entries: this.layout_entries_radix_scan,
    })
    engine_ctx.cache.bindGroupLayouts.radixScan = this.bindGroupLayoutRadixScan

    console.log('numInts', numInts)

    this.pipeline_radix_scan = GetPipelineRadixScan(
      engine_ctx,
      this.num_groups_radix_scan,
      this.count,
    )
    this.pipeline_wg_scan = GetPipelineWGScan(
      engine_ctx,
      this.num_groups_radix_scan,
      this.count,
    )
    this.pipeline_radix_scan_sync = GetPipelineRadixScanSync(
      engine_ctx,
      this.num_groups_radix_scan,
      this.count,
    )

    this.bind_group_radix_scan = new Array(2)
    this.bind_group_radix_scatter = new Array(2)
    for (let i = 0; i < 2; i++) {
      const group_entries_radix_scan: GPUBindGroupEntry[] = [
        {
          binding: 0,
          resource: {
            buffer: this.buf_constant_radix_scan,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.buf_tmp[i]!,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.prefix_wg_buffer,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.buf_tmp[1 - i]!,
          },
        },
      ]

      this.bind_group_radix_scan[i] = engine_ctx.device.createBindGroup({
        layout: this.bindGroupLayoutRadixScan,
        entries: group_entries_radix_scan,
      })
    }
  }

  sort(hInput: GPUBuffer, numIntersections?: number) {
    if (!this.engine_ctx.queue) throw new Error('this.engine_ctx.queue is null')
    if (!this.engine_ctx.device) {
      throw new Error('this.engine_ctx.device is null')
    }

    if (
      hInput.size !==
      this.count * 4 * this.numInts
      // || hInput.size !== this.buf_data.size
    ) {
      throw new Error('Input buffer size does not match the size of the sorter')
    }

    const roundedSize =
      numIntersections !== undefined ? numIntersections * 4 * 2 : hInput.size
    const roundedOffset =
      numIntersections !== undefined ? hInput.size - roundedSize : 0

    {
      const commandEncoder = this.engine_ctx.device.createCommandEncoder()

      commandEncoder.clearBuffer(this.buf_tmp[0]!)

      commandEncoder.copyBufferToBuffer(
        hInput,
        0,
        this.buf_tmp[0]!,
        roundedOffset,
        roundedSize,
      )

      this.renderer.timestamp(commandEncoder, 'copy input')

      const cmdBuf = commandEncoder.finish()
      this.engine_ctx.queue.submit([cmdBuf])
    }

    const numActiveBlocks =
      numIntersections !== undefined
        ? Math.ceil(numIntersections / WORKGROUP_SIZE)
        : this.num_groups_radix_scan

    for (let i = this.bitsOffset; i < this.bits; i += NUM_BITS_IN_CHUNK) {
      {
        const uniform = new Int32Array(4)
        uniform[0] = this.count
        uniform[1] = i
        uniform[2] = numActiveBlocks

        this.engine_ctx.queue.writeBuffer(
          this.buf_constant_radix_scan,
          0,
          uniform.buffer,
          uniform.byteOffset,
          uniform.byteLength,
        )
      }

      {
        const uniform = new Int32Array(4)
        uniform[0] = this.count
        uniform[1] = i
        uniform[2] = numActiveBlocks

        this.engine_ctx.queue.writeBuffer(
          this.buf_constant_radix_scatter,
          0,
          uniform.buffer,
          uniform.byteOffset,
          uniform.byteLength,
        )
      }

      const commandEncoder = this.engine_ctx.device.createCommandEncoder()

      // clear Buffer
      commandEncoder.clearBuffer(this.prefix_wg_buffer)

      const j = i % (NUM_BITS_IN_CHUNK * 2) < NUM_BITS_IN_CHUNK ? 0 : 1
      {
        const passEncoder = commandEncoder.beginComputePass()
        passEncoder.setPipeline(this.pipeline_radix_scan)
        passEncoder.setBindGroup(0, this.bind_group_radix_scan[j]!)

        passEncoder.dispatchWorkgroups(this.num_groups_radix_scan)
        // passEncoder.dispatchWorkgroups(numActiveBlocks)
        passEncoder.end()
        this.renderer.timestamp(commandEncoder, `scan-${i}`)
      }

      {
        const passEncoder = commandEncoder.beginComputePass()
        passEncoder.setPipeline(this.pipeline_wg_scan)
        passEncoder.setBindGroup(0, this.bind_group_radix_scan[j]!)

        passEncoder.dispatchWorkgroups(1, 1, 1)
        passEncoder.end()
        this.renderer.timestamp(commandEncoder, `wg-sum-${i}`)
      }

      {
        const passEncoder = commandEncoder.beginComputePass()
        passEncoder.setPipeline(this.pipeline_radix_scan_sync)
        passEncoder.setBindGroup(0, this.bind_group_radix_scan[j]!)

        passEncoder.dispatchWorkgroups(numActiveBlocks)
        // passEncoder.dispatchWorkgroups(numActiveBlocks + 15000)
        passEncoder.end()
        this.renderer.timestamp(commandEncoder, `sync-${i}`)
      }

      const cmdBuf = commandEncoder.finish()
      this.engine_ctx.queue.submit([cmdBuf])
    }

    {
      // const j = this.bits % 2
      const j = this.bits % (NUM_BITS_IN_CHUNK * 2) < NUM_BITS_IN_CHUNK ? 0 : 1
      const commandEncoder = this.engine_ctx.device.createCommandEncoder()
      this.renderer.timestamp(commandEncoder, 'sort')

      commandEncoder.copyBufferToBuffer(
        this.buf_tmp[j]!,
        0,
        hInput,
        // this.buf_data,
        0,
        this.buf_tmp[j]!.size,
      )

      this.renderer.timestamp(commandEncoder, 'copy back')

      const cmdBuf = commandEncoder.finish()
      this.engine_ctx.queue.submit([cmdBuf])
    }

    return {
      values: hInput,
    }
  }
}
