const num_tiles: i32 = 1;


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
@group(0) @binding(6) var<storage, read_write> intersection_offsets: array<u32>;


@compute @workgroup_size(1)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_invocation_index: u32,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {  

  var sum: u32 = 0u;
  var tmp: u32 = 0u;

  for (var i = 0; i < num_tiles; i++) {
    tmp = intersection_offsets[i];
    intersection_offsets[i] = sum;
    sum += tmp;
  }
}


