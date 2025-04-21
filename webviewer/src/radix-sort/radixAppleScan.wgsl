const workgroup_size = 1;
const num_bits_in_chunk = 1;
const two_pow_bits = 16;
const num_workgroups_global = 1;
const last_wg_padding = 1;
const histogram_size = 512;


struct Params {
  count: i32,
  bit: u32,
  num_active_blocks: i32,
};


struct DataInner {
  key: u32,
  gauss_id: u32,
};

@group(0) @binding(0) var<uniform> u_params: Params;
@group(0) @binding(1) var<storage, read> b_input : array<u32>; // array<DataInner>;
@group(0) @binding(2) var<storage, read_write> prefixes_wg : array<u32>;
@group(0) @binding(3) var<storage, read_write> b_output : array<DataInner>;


var<workgroup> sum_buf : array<atomic<u32>, histogram_size>;


@compute @workgroup_size(256)
fn main(
  @builtin(local_invocation_id) LocalInvocationID: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>,
) {
  let threadIdx = i32(LocalInvocationID.x);
  let blockIdx = i32(workgroup_id.x);
  let num_blocks = i32(num_workgroups.x);

  let idx_global = threadIdx + blockIdx * workgroup_size;


  if (idx_global < u_params.count) {
    let input = b_input[idx_global * 2]; // Key

    let bits = bitExtracted(input, u_params.bit, u32(num_bits_in_chunk));
    

    if(blockIdx == num_workgroups_global - 1) {
      atomicAdd(&sum_buf[(workgroup_size - 1) * two_pow_bits + i32(bits) - last_wg_padding], 1u);
    } else {
      atomicAdd(&sum_buf[(workgroup_size - 1) * two_pow_bits + i32(bits)], 1u);
    }
  }


  workgroupBarrier();


  if(threadIdx < two_pow_bits) {
    if(blockIdx == num_workgroups_global - 1) {

      let value = atomicLoad(&sum_buf[(workgroup_size - 1) * two_pow_bits + i32(threadIdx) - last_wg_padding]);

      prefixes_wg[blockIdx + num_workgroups_global * i32(threadIdx)] = value;


    } else {

      let value = atomicLoad(&sum_buf[(workgroup_size - 1) * two_pow_bits + i32(threadIdx)]);

      prefixes_wg[blockIdx + num_workgroups_global * i32(threadIdx)] = value;


    }
  }
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
