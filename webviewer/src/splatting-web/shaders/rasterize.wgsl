const sh_degree = 3;
const n_sh_coeffs = 16;
const intersection_array_length = 1;

const BLOCK_X = 16u;
const BLOCK_Y = 16u;

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

struct AuxData {
  num_intersections: u32,
  num_visible_gaussians: u32,
  min_depth: i32,
  max_depth: i32,
}

struct TileDepthKey {
  key: u32,
  gauss_id: u32,
}


var<workgroup> start_offset_shared: i32;
var<workgroup> end_offset_shared: i32;



@group(0) @binding(0) var color_buffer: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read_write> gauss_data: array<GaussData>;
@group(0) @binding(3) var<storage, read_write> intersection_keys: array<TileDepthKey>;
@group(0) @binding(4) var<storage, read_write> intersection_offsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> auxData: AuxData;


@compute @workgroup_size(4, 8)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_invocation_index: u32,
) {

  let global_id_x = i32(global_id.x) * 4;
  let global_id_y = i32(global_id.y) * 2;

  let u = f32(global_id_x) / f32(uniforms.screen_size.x);
  let v = f32(global_id_y) / f32(uniforms.screen_size.y);

  let num_tiles_x = u32(uniforms.screen_size.x) / BLOCK_X;
  let num_tiles_y = u32(uniforms.screen_size.y) / BLOCK_Y;

  let tile_index_x = workgroup_id.x;
  let tile_index_y = workgroup_id.y;

  let tile_index = u32(tile_index_x + tile_index_y * num_tiles_x);


  let num_groups = num_tiles_x * num_tiles_y;

  let tile_id = i32(
    workgroup_id.x +
    workgroup_id.y * num_workgroups.x +
    workgroup_id.z * num_workgroups.x * num_workgroups.y
  );

  let thread_idx = i32(local_invocation_index);


  let base_offset = u32(intersection_array_length) - auxData.num_intersections;
  
  if(thread_idx == 0) {
    start_offset_shared = i32(intersection_offsets[tile_id]);
    end_offset_shared = i32(intersection_offsets[tile_id + 1]);
  }

  var transparency_00 = 1.0;
  var transparency_10 = 1.0;
  var transparency_20 = 1.0;
  var transparency_30 = 1.0;
  var transparency_01 = 1.0;
  var transparency_11 = 1.0;
  var transparency_21 = 1.0;
  var transparency_31 = 1.0;

  var color_00 = vec3(0.0);
  var color_10 = vec3(0.0);
  var color_20 = vec3(0.0);
  var color_30 = vec3(0.0);
  var color_01 = vec3(0.0);
  var color_11 = vec3(0.0);
  var color_21 = vec3(0.0);
  var color_31 = vec3(0.0);


  let start_offset_local = workgroupUniformLoad(&start_offset_shared);
  let end_offset_local = workgroupUniformLoad(&end_offset_shared);

  for (var idx = start_offset_local; idx < end_offset_local; idx++) {
    let i = base_offset + u32(idx);

    let gauss_id = intersection_keys[i].gauss_id;

    let conic = gauss_data[gauss_id].conic;
    let opacity = gauss_data[gauss_id].opacity;

    let xy_gauss = vec2(gauss_data[gauss_id].uv.x * f32(uniforms.screen_size.x), gauss_data[gauss_id].uv.y * f32(uniforms.screen_size.y));
    let xy_pixel = vec2(u * f32(uniforms.screen_size.x), v * f32(uniforms.screen_size.y));

    let distance_00 = xy_gauss - xy_pixel;
    let distance_10 = distance_00 - vec2(1.0, 0.0);
    let distance_20 = distance_00 - vec2(2.0, 0.0);
    let distance_30 = distance_00 - vec2(3.0, 0.0);
    let distance_01 = distance_00 - vec2(0.0, 1.0);
    let distance_11 = distance_00 - vec2(1.0, 1.0);
    let distance_21 = distance_00 - vec2(2.0, 1.0);
    let distance_31 = distance_00 - vec2(3.0, 1.0);

    let power_00 = -0.5 * (conic.x * distance_00.x * distance_00.x + conic.z * distance_00.y * distance_00.y) - conic.y * distance_00.x * distance_00.y;
    let power_10 = -0.5 * (conic.x * distance_10.x * distance_10.x + conic.z * distance_10.y * distance_10.y) - conic.y * distance_10.x * distance_10.y;
    let power_20 = -0.5 * (conic.x * distance_20.x * distance_20.x + conic.z * distance_20.y * distance_20.y) - conic.y * distance_20.x * distance_20.y;
    let power_30 = -0.5 * (conic.x * distance_30.x * distance_30.x + conic.z * distance_30.y * distance_30.y) - conic.y * distance_30.x * distance_30.y;
    let power_01 = -0.5 * (conic.x * distance_01.x * distance_01.x + conic.z * distance_01.y * distance_01.y) - conic.y * distance_01.x * distance_01.y;
    let power_11 = -0.5 * (conic.x * distance_11.x * distance_11.x + conic.z * distance_11.y * distance_11.y) - conic.y * distance_11.x * distance_11.y;
    let power_21 = -0.5 * (conic.x * distance_21.x * distance_21.x + conic.z * distance_21.y * distance_21.y) - conic.y * distance_21.x * distance_21.y;
    let power_31 = -0.5 * (conic.x * distance_31.x * distance_31.x + conic.z * distance_31.y * distance_31.y) - conic.y * distance_31.x * distance_31.y;

    let alpha_00 = min(opacity * exp(power_00), 0.99);
    let alpha_10 = min(opacity * exp(power_10), 0.99);
    let alpha_20 = min(opacity * exp(power_20), 0.99);
    let alpha_30 = min(opacity * exp(power_30), 0.99);
    let alpha_01 = min(opacity * exp(power_01), 0.99);
    let alpha_11 = min(opacity * exp(power_11), 0.99);
    let alpha_21 = min(opacity * exp(power_21), 0.99);
    let alpha_31 = min(opacity * exp(power_31), 0.99);

    color_00 += gauss_data[gauss_id].color * (alpha_00 * transparency_00);
    color_10 += gauss_data[gauss_id].color * (alpha_10 * transparency_10);
    color_20 += gauss_data[gauss_id].color * (alpha_20 * transparency_20);
    color_30 += gauss_data[gauss_id].color * (alpha_30 * transparency_30);
    color_01 += gauss_data[gauss_id].color * (alpha_01 * transparency_01);
    color_11 += gauss_data[gauss_id].color * (alpha_11 * transparency_11);
    color_21 += gauss_data[gauss_id].color * (alpha_21 * transparency_21);
    color_31 += gauss_data[gauss_id].color * (alpha_31 * transparency_31);

    transparency_00 *= (1.0 - alpha_00);
    transparency_10 *= (1.0 - alpha_10);
    transparency_20 *= (1.0 - alpha_20);
    transparency_30 *= (1.0 - alpha_30);
    transparency_01 *= (1.0 - alpha_01);
    transparency_11 *= (1.0 - alpha_11);
    transparency_21 *= (1.0 - alpha_21);
    transparency_31 *= (1.0 - alpha_31);

    let transparency_0 = max(max(transparency_00, transparency_10), max(transparency_20, transparency_30));
    let transparency_1 = max(max(transparency_01, transparency_11), max(transparency_21, transparency_31));

    if(max(transparency_0, transparency_1) < 1.0 / 255.0) {
      break;
    }
  }


  // textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(0, 0), vec4(color_00, max(1.0, transparency_00)));
  // textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(1, 0), vec4(color_10, max(1.0, transparency_10)));
  // textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(2, 0), vec4(color_20, max(1.0, transparency_20)));
  // textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(3, 0), vec4(color_30, max(1.0, transparency_30)));
  // textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(0, 1), vec4(color_01, max(1.0, transparency_01)));
  // textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(1, 1), vec4(color_11, max(1.0, transparency_11)));
  // textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(2, 1), vec4(color_21, max(1.0, transparency_21)));
  // textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(3, 1), vec4(color_31, max(1.0, transparency_31)));
  textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(0, 0), vec4(color_00, 1.0));
  textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(1, 0), vec4(color_10, 1.0));
  textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(2, 0), vec4(color_20, 1.0));
  textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(3, 0), vec4(color_30, 1.0));
  textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(0, 1), vec4(color_01, 1.0));
  textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(1, 1), vec4(color_11, 1.0));
  textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(2, 1), vec4(color_21, 1.0));
  textureStore(color_buffer, vec2(global_id_x, global_id_y) + vec2(3, 1), vec4(color_31, 1.0));
}
