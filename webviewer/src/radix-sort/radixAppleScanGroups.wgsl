const workgroup_size = 1;
const workgroup_size_2x = 1;
// const workgroup_cache_size = 1;
const num_bits_in_chunk = 1;
const two_pow_bits = 16;
const num_workgroups_global = 1;
const num_workgroups_global_2x = 1;
const num_inputs = 1;
const last_wg_padding = 1;
const histogram_size = 512;
const num_wg_prefixes = 1;

struct Params {
  count: i32,
  bit: u32,
};

struct Prefixes {
  s_buf0: i32,
  s_buf1: i32,
  s_buf2: i32,
  s_buf3: i32,
}

struct DataInner {
  key: u32,
  gauss_id: u32,
};

@group(0) @binding(0) var<uniform> u_params: Params;
@group(0) @binding(1) var<storage, read> b_input : array<u32>; // array<DataInner>;
@group(0) @binding(2) var<storage, read_write> prefixes_wg : array<u32>;
@group(0) @binding(3) var<storage, read_write> b_output : array<DataInner>;


var<workgroup> sum_buf : array<atomic<u32>, workgroup_size>;


@compute @workgroup_size(256, 1, 1)
fn main(
  @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
  let threadIdx = i32(LocalInvocationID.x);
  let blockIdx = i32(workgroup_id.x);
  let num_blocks = i32(num_workgroups.x);


  let elements_per_thread = num_wg_prefixes / workgroup_size;

  // TODO: Make sure this array is the correct size while also being below the register size limit
  var sum_prev_groups_sum: array<u32, 4000>;
  // var sum_prev_groups_sum: array<u32, elements_per_thread>;

  var sum = 0u;
  var tmp = 0u;

  let start = threadIdx * elements_per_thread;

      
  for(var i = 0; i < elements_per_thread; i++) {
    tmp = prefixes_wg[start + i];

    sum_prev_groups_sum[i] = sum;
    
    sum += tmp;
  }


  // Atomically add the sum to all consecutive partial sums
  for(var i = threadIdx; i < workgroup_size; i++) {
    atomicAdd(&sum_buf[i], sum);
  }

  workgroupBarrier();


  var partial_group_sum = 0u;

  if(threadIdx > 0) {
    partial_group_sum = atomicLoad(&sum_buf[threadIdx - 1]);
  }


  for(var i = 0; i < elements_per_thread; i++) {
    let value_local = sum_prev_groups_sum[i];

    prefixes_wg[start + i] = partial_group_sum + value_local;
  }
}

