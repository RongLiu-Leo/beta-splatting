@group(0) @binding(0) var screen_sampler : sampler;
@group(0) @binding(1) var color_buffer : texture_2d<f32>;

struct VertexOutput {
  @builtin(position) Position : vec4<f32>,
  @location(0) TexCoord : vec2<f32>,
}

@vertex
fn vert_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput
{
  let positions = array<vec2<f32>, 6 > (
    vec2 < f32 > (1.0, 1.0),
    vec2 < f32 > (1.0, -1.0),
    vec2 < f32 > (-1.0, -1.0),
    vec2 < f32 > (1.0, 1.0),
    vec2 < f32 > (-1.0, -1.0),
    vec2 < f32 > (-1.0, 1.0)
  );

  let texCoords = array<vec2 < f32>, 6 > (
    vec2 < f32 > (1.0, 0.0),
    vec2 < f32 > (1.0, 1.0),
    vec2 < f32 > (0.0, 1.0),
    vec2 < f32 > (1.0, 0.0),
    vec2 < f32 > (0.0, 1.0),
    vec2 < f32 > (0.0, 0.0)
  );

  var output : VertexOutput;
  output.Position = vec4<f32>(positions[vertex_index], 0.0, 1.0);
  output.TexCoord = texCoords[vertex_index];
  return output;
}

@fragment
fn frag_main(@location(0) TexCoord : vec2 < f32>) -> @location(0) vec4 <f32>
{
  return textureSample(color_buffer, screen_sampler, TexCoord);
}
