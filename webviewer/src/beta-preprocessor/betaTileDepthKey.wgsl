const item_per_thread: i32 = 1;
const num_quads_unpaddded: i32 = 1;
const BLOCK_X = 16;
const BLOCK_Y = 16;


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

const sb_num_primitives = 2;
struct PointInput {
  @location(0) position: vec3<f32>,
  @location(1) log_scale: vec3<f32>,
  @location(2) rot: vec4<f32>,
  @location(3) c0: vec3<f32>,
  @location(4) opacity_logit: f32,
  @location(5) beta: f32,
  sb_params: array<vec3<f32>, sb_num_primitives * 2>, // Each primitive has [r,g,b] and [theta,phi,beta]
};

struct PreProcessedPoint {
  id: i32,
  radii: i32,  // Also signals whether this Beta is in frustum.
  depth: f32,
  tiles_touched: u32,
  cum_tiles_touched: u32,
  uv: vec2<f32>,
  conic: vec3<f32>,
  color: vec3<f32>,
  opacity: f32,
  beta: f32,
}

struct TileDepthKey {
  key: u32,
  beta_id: u32,
}

struct AuxData {
  num_intersections: u32,
  num_visible_betas: u32,
  min_depth: i32,
  max_depth: i32,
}


@group(0) @binding(0) var<storage, read> points: array<PointInput>;
@group(0) @binding(1) var<uniform> uniforms: Uniforms;
@group(0) @binding(2) var<storage, read_write> pre_processed: array<PreProcessedPoint>;
// Total length is num_primitive-tile-intersections for each primitive.
@group(0) @binding(3) var<storage, read_write> keys: array<TileDepthKey>;
@group(0) @binding(5) var<storage, read_write> auxData: AuxData;
@group(0) @binding(6) var<storage, read_write> intersection_offsets: array<u32>;


@compute @workgroup_size(256)
fn main(
  @builtin(global_invocation_id) global_id: vec3<u32>,
  @builtin(local_invocation_id) local_id: vec3<u32>,
  @builtin(local_invocation_index) local_invocation_index: u32,
  @builtin(workgroup_id) workgroup_id: vec3<u32>,
  @builtin(num_workgroups) num_workgroups: vec3<u32>
) {
  let thread_idx = i32(local_id.x);
  let block_idx = i32(workgroup_id.x);
  let workgroup_size = 256;

  let idx_global = thread_idx + block_idx * workgroup_size;

  if (idx_global >= num_quads_unpaddded) {
    return;
  }

  let num_tiles: vec2<u32> = uniforms.screen_size / vec2(16u);

  let i = idx_global;

  if(i >= num_quads_unpaddded) {
    return;
  }

  let primitive = pre_processed[i];

  if (primitive.radii == 0) {
    return;
  }


  // Take min and max depth to normalize depth to uint32.
  let min_depth = auxData.min_depth;
  let max_depth = auxData.max_depth;

  // let normalized_depth = ((pre_processed[i].depth) - f32(min_depth)) / f32(max_depth - min_depth);
  let normalized_depth = ((pre_processed[i].depth * 1000000.0) - f32(min_depth)) / f32(max_depth - min_depth);

  let depth_uint = u32(clamp(
    normalized_depth,
    0.0,
    1.0
  ) * (pow(2.0, 16.0) - 1.0));


  let rect: vec4<u32> = getRect(
    primitive.uv,
    primitive.radii,
    vec2<u32>(u32(uniforms.screen_size.x), u32(uniforms.screen_size.y))
  );

  let rect_min = vec2<i32>(i32(rect.x), i32(rect.y));
  let rect_max = vec2<i32>(i32(rect.z), i32(rect.w));


  // For each tile that the bounding rect overlaps, emit a 
  // key/value pair. The key is |  tile ID  |      depth      |,
  // and the value is the ID of the primitive. Sorting the values 
  // with this key yields primitive IDs in a list, such that they
  // are first sorted by tile and then by depth. 

  var off = primitive.cum_tiles_touched;

  for (var y: i32 = rect_min.y; y < rect_max.y; y++) {
    for (var x: i32 = rect_min.x; x < rect_max.x; x++) {

      let tile_index = u32(x) + u32(y) * num_tiles.x;
      
      let first_16_bits_tile_index = tile_index & 0xFFFFu;
      let first_16_bits_depth = depth_uint & 0xFFFFu;

      let concatenated_value = (first_16_bits_tile_index << 16u) | first_16_bits_depth;

      keys[off].key = concatenated_value;
      keys[off].gauss_id = u32(primitive.id);

      off++;
    }
  }
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