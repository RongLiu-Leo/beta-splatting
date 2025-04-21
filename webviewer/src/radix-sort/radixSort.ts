import { EngineContext } from './EngineContext.js'

const WORKGROUP_SIZE = 256
// const WORKGROUP_SIZE = 64
const WORKGROUP_SIZE_2x = WORKGROUP_SIZE * 2

function get_shader_radix_scan(workgroup_size = WORKGROUP_SIZE) {
  const workgroup_size_2x = workgroup_size * 2

  return `
struct Params {
  count: i32,
  bit: u32
};

@group(0) @binding(0)
var<uniform> uParams: Params;

@group(0) @binding(1)
var<storage, read> bInput : array<i32>;

@group(0) @binding(2)
var<storage, read_write> bOutput1 : array<i32>;

@group(0) @binding(3)
var<storage, read_write> bOutput2 : array<i32>;

@group(0) @binding(4)
var<storage, read_write> bWGCounter : atomic<i32>;

@group(0) @binding(5)
var<storage, read_write> bWGState : array<atomic<i32>>;

var<workgroup> s_workgroup_idx : i32;
var<workgroup> s_inclusive_prefix1 : i32;
var<workgroup> s_inclusive_prefix2 : i32;
var<workgroup> s_buf1 : array<i32, ${workgroup_size_2x}>;
var<workgroup> s_buf2 : array<i32, ${workgroup_size_2x}>;

@compute @workgroup_size(${workgroup_size}, 1, 1)
fn main(@builtin(local_invocation_id) LocalInvocationID : vec3<u32>) {
    let threadIdx = i32(LocalInvocationID.x); 

    if (threadIdx == 0) {
        s_workgroup_idx = atomicAdd(&bWGCounter, 1);
    }

    workgroupBarrier();

    let blockIdx = s_workgroup_idx;

    var i = threadIdx + blockIdx * ${workgroup_size_2x};

    if (i < uParams.count) {
        let input = bInput[i];
        let pred = (input & (1 << uParams.bit)) != 0;
        s_buf1[threadIdx] = select(1,0,pred);
        s_buf2[threadIdx] = select(0,1,pred);
    }

    i = threadIdx + ${workgroup_size} + blockIdx * ${workgroup_size_2x};

    if (i < uParams.count) {
        let input = bInput[i];
        let pred = (input & (1 << uParams.bit)) != 0;
        s_buf1[threadIdx + ${workgroup_size}] = select(1, 0, pred);
        s_buf2[threadIdx + ${workgroup_size}] = select(0, 1, pred);
    }

    workgroupBarrier();

    var half_size_group = 1;
    var size_group = 2;

    while(half_size_group <= ${workgroup_size}) {
        let gid = threadIdx / half_size_group;
        let tid = gid * size_group + half_size_group + threadIdx % half_size_group;
        i = tid + blockIdx * ${workgroup_size_2x};
        if (i < uParams.count)
        {
            s_buf1[tid] = s_buf1[gid * size_group + half_size_group - 1] + s_buf1[tid];
            s_buf2[tid] = s_buf2[gid * size_group + half_size_group - 1] + s_buf2[tid];
        }
        half_size_group = half_size_group << 1;
        size_group = size_group << 1;
        workgroupBarrier();
    }

    s_inclusive_prefix1 = 0;
    s_inclusive_prefix2 = 0;

    if (threadIdx == 0) {
        atomicStore(&bWGState[blockIdx * 5 + 1],  s_buf1[${workgroup_size_2x} - 1]);
        atomicStore(&bWGState[blockIdx * 5 + 2],  s_buf2[${workgroup_size_2x} - 1]);
        atomicStore(&bWGState[blockIdx * 5], 1);

        var j = blockIdx;
        while(j > 0)
        {
            j--;
            var state = 0;
            while(state<1)
            {
                state = atomicLoad(&bWGState[j * 5]);
            }

            if (state==2)
            {
                s_inclusive_prefix1 += atomicLoad(&bWGState[j * 5 + 3]);
                s_inclusive_prefix2 += atomicLoad(&bWGState[j * 5 + 4]);
                break;
            }
            else
            {
                s_inclusive_prefix1 += atomicLoad(&bWGState[j * 5 + 1]);
                s_inclusive_prefix2 += atomicLoad(&bWGState[j * 5 + 2]);
            }
        }

        atomicStore(&bWGState[blockIdx * 5 + 3], s_buf1[${workgroup_size_2x} - 1] + s_inclusive_prefix1);
        atomicStore(&bWGState[blockIdx * 5 + 4], s_buf2[${workgroup_size_2x} - 1] + s_inclusive_prefix2);
        atomicStore(&bWGState[blockIdx * 5], 2);
    }

    workgroupBarrier();

    i = threadIdx + blockIdx * ${workgroup_size_2x};
    if (i < uParams.count)
    {
        bOutput1[i] = s_buf1[threadIdx] + s_inclusive_prefix1;
        bOutput2[i] = s_buf2[threadIdx] + s_inclusive_prefix2;
    }

    i = threadIdx + ${workgroup_size} + blockIdx * ${workgroup_size_2x};
    if (i < uParams.count)
    {
        bOutput1[i] = s_buf1[threadIdx + ${workgroup_size}] + s_inclusive_prefix1;
        bOutput2[i] = s_buf2[threadIdx + ${workgroup_size}] + s_inclusive_prefix2;
    }
}
`
}

function GetPipelineRadixScan(engine_ctx: EngineContext) {
  if (!engine_ctx.device) throw new Error('engine_ctx.device is null')

  const shaderModule = engine_ctx.device.createShaderModule({
    code: get_shader_radix_scan(),
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

function get_shader_radix_scatter(workgroup_size = WORKGROUP_SIZE) {
  return `
@group(0) @binding(0)
var<uniform> uCount: i32;

@group(0) @binding(1)
var<storage, read> bInput : array<i32>;

@group(0) @binding(2)
var<storage, read> bIndices1 : array<i32>;

@group(0) @binding(3)
var<storage, read> bIndices2 : array<i32>;

@group(0) @binding(4)
var<storage, read_write> bOutput : array<i32>;

@compute @workgroup_size(${workgroup_size},1,1)
fn main(@builtin(global_invocation_id) GlobalInvocationID : vec3<u32>) {
    let idx = i32(GlobalInvocationID.x);

    if (idx >= uCount) {
        return;
    }

    let value = bInput[idx];

    if ((idx == 0 && bIndices1[idx] > 0) || (idx > 0 && bIndices1[idx] > bIndices1[idx - 1]))
    {
        bOutput[bIndices1[idx] - 1] = value;
    }
    else
    {
        let count0 = bIndices1[uCount - 1];
        bOutput[count0 + bIndices2[idx] - 1] = value;
    }
}
`
}

function GetPipelineRadixScatter(engine_ctx: EngineContext) {
  if (!engine_ctx.device) throw new Error('engine_ctx.device is null')

  const shaderModule = engine_ctx.device.createShaderModule({
    code: get_shader_radix_scatter(),
  })
  const bindGroupLayouts = [engine_ctx.cache.bindGroupLayouts.radixScatter]
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

function getRandomInt(max: number) {
  return Math.floor(Math.random() * max)
}

export async function test() {
  const engine_ctx = new EngineContext()
  await engine_ctx.initialize()

  // const count = 2 ** 16
  const count = 2 ** 7
  const num_groups_radix_scan = Math.floor(
    (count + WORKGROUP_SIZE_2x - 1) / WORKGROUP_SIZE_2x,
  )
  const max_value = 10000
  const hInput = new Int32Array(count)
  for (let i = 0; i < count; i++) {
    hInput[i] = getRandomInt(max_value)
  }

  const hReference = new Int32Array(count)
  hReference.set(hInput)
  hReference.sort()

  const buf_data = engine_ctx.createBuffer0(
    count * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  )

  if (!engine_ctx.queue) throw new Error('engine_ctx.queue is null')

  engine_ctx.queue.writeBuffer(
    buf_data,
    0,
    hInput.buffer,
    hInput.byteOffset,
    hInput.byteLength,
  )

  const buf_tmp = new Array(2)
  buf_tmp[0] = engine_ctx.createBuffer0(
    count * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  )
  buf_tmp[1] = engine_ctx.createBuffer0(
    count * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  )

  const buf_constant_radix_scan = engine_ctx.createBuffer0(
    16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  )
  const buf_indices1 = engine_ctx.createBuffer0(
    count * 4,
    GPUBufferUsage.STORAGE,
  )
  const buf_indices2 = engine_ctx.createBuffer0(
    count * 4,
    GPUBufferUsage.STORAGE,
  )
  const buf_workgroup_counter = engine_ctx.createBuffer0(
    4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  )
  const buf_workgroup_state = engine_ctx.createBuffer0(
    num_groups_radix_scan * 5 * 4,
    GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  )

  const buf_constant_radix_scatter = engine_ctx.createBuffer0(
    16,
    GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  )

  const buf_download = engine_ctx.createBuffer0(
    count * 4,
    GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  )

  const layout_entries_radix_scan: globalThis.GPUBindGroupLayoutEntry[] = [
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
  ]

  if (!engine_ctx.device) throw new Error('engine_ctx.device is null')

  const bindGroupLayoutRadixScan = engine_ctx.device.createBindGroupLayout({
    entries: layout_entries_radix_scan,
  })
  engine_ctx.cache.bindGroupLayouts.radixScan = bindGroupLayoutRadixScan

  const pipeline_radix_scan = GetPipelineRadixScan(engine_ctx)

  const layout_entries_radix_scatter: globalThis.GPUBindGroupLayoutEntry[] = [
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
        type: 'read-only-storage',
      },
    },
    {
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: 'read-only-storage',
      },
    },
    {
      binding: 4,
      visibility: GPUShaderStage.COMPUTE,
      buffer: {
        type: 'storage',
      },
    },
  ]

  const bindGroupLayoutRadixScatter = engine_ctx.device.createBindGroupLayout({
    entries: layout_entries_radix_scatter,
  })
  engine_ctx.cache.bindGroupLayouts.radixScatter = bindGroupLayoutRadixScatter

  const pipeline_radix_scatter = GetPipelineRadixScatter(engine_ctx)

  const bind_group_radix_scan = new Array(2)
  const bind_group_radix_scatter = new Array(2)
  for (let i = 0; i < 2; i++) {
    const group_entries_radix_scan = [
      {
        binding: 0,
        resource: {
          buffer: buf_constant_radix_scan,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: buf_tmp[i],
        },
      },
      {
        binding: 2,
        resource: {
          buffer: buf_indices1,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: buf_indices2,
        },
      },
      {
        binding: 4,
        resource: {
          buffer: buf_workgroup_counter,
        },
      },
      {
        binding: 5,
        resource: {
          buffer: buf_workgroup_state,
        },
      },
    ]

    bind_group_radix_scan[i] = engine_ctx.device.createBindGroup({
      layout: bindGroupLayoutRadixScan,
      entries: group_entries_radix_scan,
    })

    const group_entries_radix_scatter = [
      {
        binding: 0,
        resource: {
          buffer: buf_constant_radix_scatter,
        },
      },
      {
        binding: 1,
        resource: {
          buffer: buf_tmp[i],
        },
      },
      {
        binding: 2,
        resource: {
          buffer: buf_indices1,
        },
      },
      {
        binding: 3,
        resource: {
          buffer: buf_indices2,
        },
      },
      {
        binding: 4,
        resource: {
          buffer: buf_tmp[1 - i],
        },
      },
    ]

    bind_group_radix_scatter[i] = engine_ctx.device.createBindGroup({
      layout: bindGroupLayoutRadixScatter,
      entries: group_entries_radix_scatter,
    })
  }

  const bits = 32
  // const bits = 14

  {
    const commandEncoder = engine_ctx.device.createCommandEncoder()
    commandEncoder.copyBufferToBuffer(buf_data, 0, buf_tmp[0], 0, count * 4)
    const cmdBuf = commandEncoder.finish()
    engine_ctx.queue.submit([cmdBuf])
  }

  for (let i = 0; i < bits; i++) {
    {
      const uniform = new Int32Array(4)
      uniform[0] = count
      uniform[1] = i

      engine_ctx.queue.writeBuffer(
        buf_constant_radix_scan,
        0,
        uniform.buffer,
        uniform.byteOffset,
        uniform.byteLength,
      )

      const group_count = new Int32Array(1)
      engine_ctx.queue.writeBuffer(
        buf_workgroup_counter,
        0,
        group_count.buffer,
        group_count.byteOffset,
        group_count.byteLength,
      )

      const group_state = new Int32Array(num_groups_radix_scan * 5)
      engine_ctx.queue.writeBuffer(
        buf_workgroup_state,
        0,
        group_state.buffer,
        group_state.byteOffset,
        group_state.byteLength,
      )
    }

    {
      const uniform = new Int32Array(4)
      uniform[0] = count
      engine_ctx.queue.writeBuffer(
        buf_constant_radix_scatter,
        0,
        uniform.buffer,
        uniform.byteOffset,
        uniform.byteLength,
      )
    }

    const commandEncoder = engine_ctx.device.createCommandEncoder()

    const j = i % 2
    {
      const passEncoder = commandEncoder.beginComputePass()
      passEncoder.setPipeline(pipeline_radix_scan)
      passEncoder.setBindGroup(0, bind_group_radix_scan[j])
      passEncoder.dispatchWorkgroups(num_groups_radix_scan, 1, 1)
      passEncoder.end()
    }

    {
      const num_groups = (count + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE
      const passEncoder = commandEncoder.beginComputePass()
      passEncoder.setPipeline(pipeline_radix_scatter)
      passEncoder.setBindGroup(0, bind_group_radix_scatter[j])
      passEncoder.dispatchWorkgroups(num_groups, 1, 1)
      passEncoder.end()
    }

    const cmdBuf = commandEncoder.finish()
    engine_ctx.queue.submit([cmdBuf])
  }

  {
    const j = bits % 2
    const commandEncoder = engine_ctx.device.createCommandEncoder()
    commandEncoder.copyBufferToBuffer(buf_tmp[j], 0, buf_data, 0, count * 4)
    commandEncoder.copyBufferToBuffer(buf_data, 0, buf_download, 0, count * 4)
    const cmdBuf = commandEncoder.finish()
    engine_ctx.queue.submit([cmdBuf])
  }

  const hOutput = new Int32Array(count)
  {
    await buf_download.mapAsync(GPUMapMode.READ)
    const buf = buf_download.getMappedRange()
    hOutput.set(new Int32Array(buf))
    buf_download.unmap()
  }

  console.log('hInput', hInput)
  console.log('hOutput', hOutput)
  console.log('hReference', hReference)

  let count_unmatch = 0

  for (let i = 0; i < count; i++) {
    if (hOutput[i] !== hReference[i]) {
      count_unmatch++
    }
  }

  console.log(`count_unmatch: ${count_unmatch}`)
}
