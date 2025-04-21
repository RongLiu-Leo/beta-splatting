const workgroup_size = 1;
const num_bits_in_chunk = 1;
const two_pow_bits = 16;
const num_workgroups_global = 1;
const histogram_size = 512;

struct Params {
  count: u32,
  bit: u32,
  num_active_blocks: i32,
};



struct DataInner {
  key: u32,
  gauss_id: u32,
};

@group(0) @binding(0) var<uniform> u_params: Params;
@group(0) @binding(1) var<storage, read> b_input : array<DataInner>;
@group(0) @binding(2) var<storage, read_write> prefixes_wg : array<u32>;
@group(0) @binding(3) var<storage, read_write> b_output : array<DataInner>;


var<workgroup> sum_buf : array<atomic<i32>, histogram_size>;


@compute @workgroup_size(256)
fn sync(
  @builtin(local_invocation_id) LocalInvocationID : vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>,
  @builtin(global_invocation_id) GlobalInvocationID : vec3<u32>
) {
  let threadIdx = u32(LocalInvocationID.x);

  // TODO: Fix this attempt to only sort the filled part of the array
  
  // let blockIdx  = i32(workgroup_id.x);
  // // TODO: Why do we need this offset? Can't get it to work with 0
  let blockIdx  = i32(workgroup_id.x) + max((num_workgroups_global - u_params.num_active_blocks), 0);
  // let blockIdx  = i32(workgroup_id.x) + max((num_workgroups_global - u_params.num_active_blocks - 15000), 0);

  // let blockIdx  = i32(workgroup_id.x) + 2;
  let num_blocks = i32(num_workgroups.x);
  let num_sub_chunks = 8;



  let idx_global = threadIdx + u32((blockIdx) * workgroup_size);
  var sum_prev_groups_sum: array<u32, two_pow_bits>;

  // TODO: Reduce num of global memory accesses, probably can move it to the bottom summation

  // 0.6 ms
  // if(threadIdx == 0u){
    for(var i = 0u; i < u32(two_pow_bits); i++) {
      sum_prev_groups_sum[i] = prefixes_wg[blockIdx + (num_workgroups_global) * i32(i)];
    }
  // }

  workgroupBarrier();

  

  let idx_1_local = threadIdx;

  if (idx_global < u_params.count) {
    let input = b_input[idx_global].key;

    let bits = bitExtracted(input, u_params.bit, u32(num_bits_in_chunk));

    atomicAdd(&sum_buf[i32(idx_1_local) * two_pow_bits + i32(bits)], 1);


    // 2.6 ms
    let factor = i32(threadIdx) / (workgroup_size / num_sub_chunks) + 1;
    let start = factor;
    let end = num_sub_chunks;

    for(var i = start; i < end; i++) {
      atomicAdd(&sum_buf[i * two_pow_bits * workgroup_size / num_sub_chunks + i32(bits)], 1);
    }
  }


  workgroupBarrier();




  var sum = 0;

  // TODO: Coalesce the access to sum_buf (requires reindexing)

  // 4.8 ms
  if(threadIdx < u32(two_pow_bits * num_sub_chunks)) {
    let factor = i32(threadIdx) % num_sub_chunks;

    let start = factor * workgroup_size / num_sub_chunks;

    let end = start + workgroup_size / num_sub_chunks;

    for (var i = start; i < end; i++) {
    
    // for (var i = 0; i < workgroup_size; i++) {
      sum += atomicLoad(&sum_buf[i * two_pow_bits + i32(threadIdx) / num_sub_chunks]);

      atomicStore(&sum_buf[i * two_pow_bits + i32(threadIdx) / num_sub_chunks], sum);
    }
    
  }



  workgroupBarrier();

  // Scatter

  if (idx_global >= u_params.count) {
    return;
  }

  var value = b_input[idx_global];

  let bits = bitExtracted(value.key, u_params.bit, u32(num_bits_in_chunk));

  let prev_sum = i32(sum_prev_groups_sum[bits]);


  let idx_scatter = atomicLoad(&sum_buf[idx_1_local * u32(two_pow_bits) + bits]) + prev_sum;

  b_output[idx_scatter - 1] = value;
}



fn bitExtracted(input: u32, start_pos: u32, num_bits: u32) -> u32 {
  // Create a mask with 'length' number of 1's
  let mask = (1u << num_bits) - 1u;
  // Shift the mask to start at the 'start' position
  let shifted_mask = mask << start_pos;
  // Bitwise AND to get the bits in the range
  let result = input & shifted_mask;
  // Shift the result back to start from position 0
  let shifted_result = result >> start_pos;
  
  return shifted_result;
}