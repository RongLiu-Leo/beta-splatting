const BLOCK_X = 16;
const BLOCK_Y = 16;
const num_quads_unpaddded: i32 = 1;
const item_per_thread: i32 = 1;
const workgroup_size = 256;


struct Uniforms {
  viewMatrix: mat4x4<f32>,
  projMatrix: mat4x4<f32>,
  camera_position: vec3<f32>,
  tan_fovx: f32,
  tan_fovy: f32,
  focal_x: f32,
  focal_y: f32,
  scale_modifier: f32,
  screen_size: vec2<u32>,
};

const sh_degree = 3;
const n_sh_coeffs = 16;
struct PointInput {
  @location(0) position: vec3<f32>,
  @location(1) log_scale: vec3<f32>,
  @location(2) rot: vec4<f32>,
  @location(3) opacity_logit: f32,
  sh: array<vec3<f32>, n_sh_coeffs>,
};

struct AuxData {
  num_intersections: atomic<u32>,
  num_visible_gaussians: atomic<u32>,
  min_depth: atomic<i32>,
  max_depth: atomic<i32>,
}

struct GaussData {
  id: i32,
  radii: i32,  // Also signals whether this Gaussian is in frustum.
  depth: f32,
  tiles_touched: u32,
  cum_tiles_touched: u32,
  uv: vec2<f32>,
  conic: vec3<f32>,
  color: vec3<f32>,
  opacity: f32,
}

struct TileDepthKey {
  key: u32,
  gauss_id: u32,
}


@group(0) @binding(0) var<storage, read> points: array<PointInput>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read_write> result: array<GaussData>;
@group(0) @binding(3) var<storage, read_write> intersection_keys: array<TileDepthKey>;
@group(0) @binding(4) var<storage, read_write> prefixes: array<u32>;
@group(0) @binding(5) var<storage, read_write> auxData: AuxData;
@group(0) @binding(6) var<storage, read_write> intersection_offsets: array<atomic<u32>>;


@compute @workgroup_size(1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_invocation_index: u32,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let thread_idx = i32(local_id.x);
  let block_idx = i32(workgroup_id.x);
  

  let idx_global = thread_idx + block_idx * workgroup_size;

  // if (idx_global >= num_quads_unpaddded) {
  //   return;
  // }

  var sum: u32 = 0u;

  for (var i = i32(global_id.x) * item_per_thread; i < (i32(global_id.x) + 1) * item_per_thread; i++) {

    // let i = idx_global;

    if(i >= num_quads_unpaddded) {
      // return;
      break;
    }

    result[i].cum_tiles_touched = sum;

    sum += result[i].tiles_touched;

    let workgroup_id = i / item_per_thread;
    prefixes[workgroup_id] = sum;

    // prefixes[block_idx] = sum;

  }
}



@compute @workgroup_size(1)
fn prefix_sum_sync(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_invocation_index: u32,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  // Calculate the global id based on the workgroup location and the local id
  let workgroup_index = i32(
    workgroup_id.x +
    workgroup_id.y * num_workgroups.x +
    workgroup_id.z * num_workgroups.x * num_workgroups.y
  );

  let tot_num_workgroups = i32(
    num_workgroups.x *
    num_workgroups.y *
    num_workgroups.z
  );

  let global_index = workgroup_index * item_per_thread + i32(local_id.x);

  var sum_of_prev_workgroups: u32 = 0u;

  // get the sum of all the previous workgroups
  for (var i = 1; i < workgroup_index + 1; i++) {
    sum_of_prev_workgroups += prefixes[i - 1];
  }


  // add the sum of all the previous workgroups to the cumsum of the current workgroup
  for (var i = 0; i < item_per_thread; i++) {
    let index = global_index + i;

    if(index >= num_quads_unpaddded) {
      break;
    }

    if(index < item_per_thread) {
      continue;
    }

    // result[index].cum_tiles_touched = u32(workgroup_index);
    result[index].cum_tiles_touched += sum_of_prev_workgroups;
  }
}


