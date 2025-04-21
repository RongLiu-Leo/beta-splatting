const BLOCK_X = 16;
const BLOCK_Y = 16;
const num_quads_unpaddded: i32 = 1;
const item_per_thread: i32 = 1;


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


@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_invocation_index: u32,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let num_tiles = vec2<u32>(
      uniforms.screen_size.x / 16u, 
      uniforms.screen_size.y / 16u
    );
  
  let thread_idx = i32(local_id.x);
  let block_idx = i32(workgroup_id.x);

  let i = thread_idx + block_idx * 256;


  if(i >= num_quads_unpaddded) {
    return;
  }

  let point: PointInput = points[i];


  let world_pos = vec4(point.position, 1.0);  
  let camera_pos = uniforms.viewMatrix * world_pos;

  if(!in_frustum(world_pos)) {
    return;
  }

  let opacity = sigmoid(point.opacity_logit);

  // TODO: Make this culling a bit less aggressive when numIntersections is lowered some other way
  if(opacity < 0.03) {
    return;
  }

  let p_hom = uniforms.projMatrix * world_pos;
  let p_w = 1.0f / (p_hom.w + 0.0000001f);

  var clip_pos = vec4<f32>(
    p_hom.x * p_w,
    p_hom.y * p_w,
    p_hom.z * p_w,
    p_hom.w * p_w
  );


  let point_uv = (clip_pos.xy * 0.5) + 0.5;


  let cov_2d = compute_cov2d(point.position, point.log_scale, point.rot);

  let det = cov_2d.x * cov_2d.z - cov_2d.y * cov_2d.y;

  if (det == 0.0f) {
    return;
  }

  let det_inv = 1.0 / det;

  let conic = vec3<f32>(
    cov_2d.z * det_inv, 
    -cov_2d.y * det_inv, 
    cov_2d.x * det_inv
  );

  // Compute extent in screen space (by finding eigenvalues of
  // 2D covariance matrix). Use extent to compute a bounding rectangle
  // of screen-space tiles that this Gaussian overlaps with. Quit if
  // rectangle covers 0 tiles. 
  let mid = 0.5 * (cov_2d.x + cov_2d.z);


  let lambda_1 = mid + sqrt(max(0.1, mid * mid - det));
  let lambda_2 = mid - sqrt(max(0.1, mid * mid - det));

  let radius_px: i32 = i32(ceil(3. * sqrt(max(lambda_1, lambda_2))));

  let rect = getRect(
    point_uv,
    radius_px,
    vec2<u32>(u32(uniforms.screen_size.x), u32(uniforms.screen_size.y))
  );

  let rect_min = rect.xy;
  let rect_max = rect.zw;

  if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0u) {return;}
  
  let color = compute_color_from_sh(world_pos.xyz, point.sh);

  
  var tiles_touched = 0u;
  
  // get x and y of every tile touched
  for (var y = rect_min.y; y < rect_max.y; y++) {
    for (var x = rect_min.x; x < rect_max.x; x++) {
      let idx = y * num_tiles.x + x;
      atomicAdd(&intersection_offsets[idx], 1u);
      tiles_touched++;
    }
  }


  
  let gauss_data = GaussData(
    i,
    radius_px,
    clip_pos.z,
    tiles_touched,
    0u,
    point_uv,
    conic,
    color,
    opacity
  );

  result[i] = gauss_data;

  atomicAdd(&auxData.num_visible_gaussians, 1u);
  atomicAdd(&auxData.num_intersections, tiles_touched);
  atomicMin(&auxData.min_depth, i32(clip_pos.z * 1000000.0));
  atomicMax(&auxData.max_depth, i32(clip_pos.z * 1000000.0));

}


fn in_frustum(world_pos: vec4<f32>) -> bool {
  let p_hom = uniforms.projMatrix * world_pos;
  let p_w = 1.0f / (p_hom.w + 0.0000001f);

  let p_proj = vec3(
    p_hom.x * p_w,
    p_hom.y * p_w,
    p_hom.z * p_w
  );

  let p_view = uniforms.viewMatrix * world_pos;

  
  if (p_view.z <= 0.2f || ((p_proj.x < -1.1 || p_proj.x > 1.1 || p_proj.y < -1.1 || p_proj.y > 1.1))) {
    return false;
  }

  return true;
}


fn compute_cov3d(log_scale: vec3<f32>, rot: vec4<f32>) -> array<f32, 6> {

  let modifier = uniforms.scale_modifier;

  let S = mat3x3<f32>(
    exp(log_scale.x) * modifier, 0., 0.,
    0., exp(log_scale.y) * modifier, 0.,
    0., 0., exp(log_scale.z) * modifier,
  );
  
  // Normalize quaternion to get valid rotation
  // let quat = rot;
  let quat = rot / length(rot);
  let r = quat.x;
  let x = quat.y;
  let y = quat.z;
  let z = quat.w;

  let R = mat3x3(
    1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y),
    2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
    2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y),
  );

  let M = S * R;
  let Sigma = transpose(M) * M;

  return array<f32, 6> (
    Sigma[0][0],
    Sigma[0][1],
    Sigma[0][2],
    Sigma[1][1],
    Sigma[1][2],
    Sigma[2][2],
  );
}


fn compute_cov2d(position: vec3<f32>, log_scale: vec3<f32>, rot: vec4<f32>) -> vec3<f32> {
  let cov3d = compute_cov3d(log_scale, rot);

  // The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.

  var t = uniforms.viewMatrix * vec4<f32>(position, 1.0);
  // let focal_x = 1.0;
  // let focal_y = 1.0;
  let focal_x = uniforms.focal_x;
  let focal_y = uniforms.focal_y;

  // Orig
  let limx = 1.3 * uniforms.tan_fovx;
  let limy = 1.3 * uniforms.tan_fovy;
  let txtz = t.x / t.z;
  let tytz = t.y / t.z;

  t.x = min(limx, max(-limx, txtz)) * t.z;
  t.y = min(limy, max(-limy, tytz)) * t.z;

  let J = mat3x3(
    focal_x / t.z,  0., -(focal_x * t.x) / (t.z * t.z),
    0., focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
    0., 0., 0.,
  );

  // this includes the transpose
  let W = mat3x3(
    uniforms.viewMatrix[0][0], uniforms.viewMatrix[1][0], uniforms.viewMatrix[2][0],
    uniforms.viewMatrix[0][1], uniforms.viewMatrix[1][1], uniforms.viewMatrix[2][1],
    uniforms.viewMatrix[0][2], uniforms.viewMatrix[1][2], uniforms.viewMatrix[2][2],
  );

  let T = W * J;

  let Vrk = mat3x3(
    cov3d[0], cov3d[1], cov3d[2],
    cov3d[1], cov3d[3], cov3d[4],
    cov3d[2], cov3d[4], cov3d[5],
  );

  var cov = transpose(T) * transpose(Vrk) * T;

  // Apply low-pass filter: every Gaussian should be at least
  // one pixel wide/high. Discard 3rd row and column.
  cov[0][0] += 0.3;
  cov[1][1] += 0.3;


  return vec3(cov[0][0], cov[0][1], cov[1][1]);
}



fn sigmoid(x: f32) -> f32 {
  // if (x >= 0.) {
  //   return 1. / (1. + exp(-x));
  // } else {
  //   let z = exp(x);
  //   return z / (1. + z);
  // }

  let z = exp(x);
  let condition = f32(x >= 0.);

  return (condition * (1. / (1. + exp(-x)))) + ((1.0 - condition) * (z / (1. + z)));
}



fn ndc2pix(v: f32, size: u32) -> f32 {
  return ((v + 1.0) * f32(size) - 1.0) * 0.5;
}



//spherical harmonic coefficients
const SH_C0 = 0.28209479177387814f;
const SH_C1 = 0.4886025119029199f;
// const SH_C2 = array(
//   1.0925484305920792f,
//   -1.0925484305920792f,
//   0.31539156525252005f,
//   -1.0925484305920792f,
//   0.5462742152960396f
// );
// const SH_C3 = array(
//   -0.5900435899266435f,
//   2.890611442640554f,
//   -0.4570457994644658f,
//   0.3731763325901154f,
//   -0.4570457994644658f,
//   1.445305721320277f,
//   -0.5900435899266435f
// );



fn compute_color_from_sh(position: vec3<f32>, sh: array<vec3<f32>, 16>) -> vec3<f32> {
  let dir = normalize(position - uniforms.camera_position);
  var result = SH_C0 * sh[0];

  // if deg > 0
  let x = dir.x;
  let y = dir.y;
  let z = dir.z;

  result = result + SH_C1 * (-y * sh[1] + z * sh[2] - x * sh[3]);

  let xx = x * x;
  let yy = y * y;
  let zz = z * z;
  let xy = x * y;
  let xz = x * z;
  let yz = y * z;

  // // if (sh_degree > 1) {
  // result = result +
  //   SH_C2[0] * xy * sh[4] +
  //   SH_C2[1] * yz * sh[5] +
  //   SH_C2[2] * (2. * zz - xx - yy) * sh[6] +
  //   SH_C2[3] * xz * sh[7] +
  //   SH_C2[4] * (xx - yy) * sh[8];
  
  // // // if (sh_degree > 2) {
  // result = result +
  //   SH_C3[0] * y * (3. * xx - yy) * sh[9] +
  //   SH_C3[1] * xy * z * sh[10] +
  //   SH_C3[2] * y * (4. * zz - xx - yy) * sh[11] +
  //   SH_C3[3] * z * (2. * zz - 3. * xx - 3. * yy) * sh[12] +
  //   SH_C3[4] * x * (4. * zz - xx - yy) * sh[13] +
  //   SH_C3[5] * z * (xx - yy) * sh[14] +
  //   SH_C3[6] * x * (xx - 3. * yy) * sh[15];

  // unconditional
  result = result + 0.5;

  return max(result, vec3<f32>(0.));
}



// TODO: Fix issue with last tile being full of intersecitons
fn getRect(uv: vec2<f32>, max_radius: i32, grid: vec2<u32>) -> vec4<u32> {
  let p: vec2<f32> = uv * vec2(f32(grid.x), f32(grid.y));
  let num_tiles_x = i32(grid.x) / BLOCK_X;
  let num_tiles_y = i32(grid.y) / BLOCK_Y;

  var rect_min: vec2<u32>;
  var rect_max: vec2<u32>;

  // Calculate rect_min
  let min_x = min(num_tiles_x, max(0, i32(p.x - f32(max_radius)) / BLOCK_X));
  rect_min.x = u32(min_x);

  let min_y = min(num_tiles_y, max(0, i32(p.y - f32(max_radius)) / BLOCK_Y));
  rect_min.y = u32(min_y);

  // Calculate rect_max
  let max_x = min(num_tiles_x, max(0, i32(p.x + f32(max_radius)) / BLOCK_X)) + 1;
  rect_max.x = u32(max_x);

  let max_y = min(num_tiles_y, max(0, i32(p.y + f32(max_radius)) / BLOCK_Y)) + 1;
  rect_max.y = u32(max_y);

  return vec4<u32>(rect_min, rect_max);
}