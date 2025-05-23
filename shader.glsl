#version 460

layout(binding = 0) uniform writeonly image2D image;

layout(local_size_x = 16, local_size_y = 16) in;

void main()
{
  ivec2 uv = ivec2(gl_GlobalInvocationID.xy);

  if (uv.x < 200 && uv.y < 200)
  {
    imageStore(image, uv, vec4(1.0));  
  }
}