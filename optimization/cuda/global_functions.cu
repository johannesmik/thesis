// Author: Johannes Mikulasch
// July 2015

#ifndef GLOBAL_FUNCTIONS_CU
#define GLOBAL_FUNCTIONS_CU

#ifndef m_depth
  #define m_depth 5
#endif
#ifndef m_ir
  #define m_ir 5
#endif

#include <cuda.h>
#include "utils.cu"
#include "intensities.cu"
#include "normal.cu"
#include "cameras.cu"
#include "lights.cu"

texture<float, cudaTextureType2D, cudaReadModeElementType> depth_sensor_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> depth_current_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> ir_sensor_tex;
texture<float, cudaTextureType2D, cudaReadModeElementType> ir_current_tex;
texture<float4, cudaTextureType2D, cudaReadModeElementType> material_current_tex;       // includes k_diff, k_spec, n

__device__ void get_current_depth_neighborhood(int2 pos, float neighborhood[m_depth][m_depth])
{
  /* returns the (m, m) current depth neighborhood around pos. */
  /* Loads texture memory into global memory (slow) */

  const int left_border = (-1 + (m_ir + 1 )/ 2);

  for (int i = 0; i < m_depth; ++i) {
    for (int j = 0; j < m_depth; ++j) {
      neighborhood[j][i] = tex2D(depth_current_tex, pos.x - left_border + i, pos.y - left_border + j);
    }
  }
}

__device__ void get_current_material_neighborhood(const int2 pos, float4 neighborhood[3][3])
{
  /* Returns the (3, 3) neighborhood of material_current_tex around pos. */

  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      neighborhood[j][i] = tex2D(material_current_tex, pos.x - 1 + i, pos.y - 1 + j);
    }
  }

}

__device__ float intensity_local(LightingModel &lightingmodel, NormalCalculator &normalmodel, int2 center_pos, int2 change_pos, float adjustment){
  // Calculate the ir around center based on 'depth_current_tex', but adjust the depth of the pixel at change_pos
  // Center_pos: screen coords
  // Change_pos: screen coords

  float depth_neighborhood[m_depth][m_depth];
  get_current_depth_neighborhood(center_pos, depth_neighborhood);
  const float z = depth_neighborhood[(m_depth + 1) / 2][(m_depth + 1) / 2];

  const int left_border = (-1 + (m_ir + 1 )/ 2);

  // Adjust
  depth_neighborhood[change_pos.y - center_pos.y + left_border][change_pos.x - center_pos.x + left_border] += adjustment;

  const float3 normal = normalmodel.normal(depth_neighborhood, center_pos);
  const float ir_return = lightingmodel.intensity(normal, pixel_to_camera(center_pos.x, center_pos.y, z));

  return ir_return;
}

inline __device__ float intensity_local(LightingModel &lightingmodel, NormalCalculator &normalmodel, int2 pos) {
  // Calculate the ir intensity at the midpoint of (m, m) depth neighborhood around pos
  return intensity_local(lightingmodel, normalmodel, pos, pos, 0.0);
}

inline __device__ float intensity_local(LightingModel &lightingmodel, NormalCalculator &normalmodel, int2 pos, float adjustment) {
  // Calculate the ir intensity around center, but adjust the depth of the pixel at center
  // pos: screen coords
  return intensity_local(lightingmodel, normalmodel, pos, pos, adjustment);
}


///////////////////////////////////

extern "C"
__global__ void energy_prime(const int lightingmodel_enum, const int normalmodel_enum,
                             const float depth_variance, const float ir_variance,
                             const float w_d, const float w_m,
                             float *depth_out, float4 *material_out) {
/*
  Calculate the energy prime using the central difference.

  Input Textures:
    - depth_current_tex
    - depth_sensor_tex
    - ir_sensor_tex
    - material_current_tex
*/

// TODO test if this works with specular model too

  // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  // Todo define variance as function input parameter or outside in the numpy function?
  const float h_depth = 0.0001;
  const float h_material = 0.01;
  //const float depth_variance = 0.001;
  //const float ir_variance = 0.001;
  //const float w_d = 1;
  //const float w_m = 50;

  float diff_d, diff_kd, diff_ks, diff_n = 0.0;

  float data_term_f, data_term_b = 0;
  float shading_constraint_f, shading_constraint_b = 0;
  float shape_prior_f, shape_prior_b = 0;
  float material_prior_f, material_prior_b = 0;

  float depth_neighborhood[m_depth][m_depth];
  get_current_depth_neighborhood(make_int2(x, y), depth_neighborhood);
  float3 normal = make_float3(0, 0, 0);

  float ir_new = 0;

  const float ir_given = tex2D(ir_sensor_tex, x, y);
  const float depth_given = tex2D(depth_sensor_tex, x, y);
  float depth_current = 0;

  PointLight light = PointLight(make_float3(0, 0, 0), 0.0);
  KinectCamera camera = KinectCamera(make_float3(0, 0, 0));

  LambertianLightingModel lambertianlightingmodel = LambertianLightingModel(light, camera);
  SpecularLightingModel specularlightingmodel = SpecularLightingModel(light, camera);

  LightingModel *lightingmodel;
  if (lightingmodel_enum == 0)
    lightingmodel = &lambertianlightingmodel;
  else if (lightingmodel_enum == 1)
    lightingmodel = &specularlightingmodel;
  lightingmodel->set_material_properties(tex2D(material_current_tex, x, y));

  NormalCalculator *normalmodel;
  NormalCross normalcross = NormalCross(camera);
  NormalPca normalpca = NormalPca(camera);
  if (normalmodel_enum == 0)
    normalmodel = &normalcross;
  else if (normalmodel_enum == 1)
    normalmodel = &normalpca;

  // Get normals around the current point, with current depth
  float3 normals[3][3] = {make_float3(0, 0, 0)};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      float depth_neighborhood[m_depth][m_depth];
      get_current_depth_neighborhood(make_int2(x - 1 + i, y - 1 + j), depth_neighborhood);
      normals[i][j] = normalmodel->normal(depth_neighborhood, make_int2(x - 1 + i, y - 1 + j));
    }
  }

  /* * * * */
  // Change in Depth (changes data term, shading constraint and shape prior)
  /* * * * */

  // forward difference of depth
  depth_neighborhood[2][2] = depth_neighborhood[2][2] + h_depth;
  depth_current = tex2D(depth_current_tex, x, y) + h_depth;
  normal = normalmodel->normal(depth_neighborhood, make_int2(x, y));

  data_term_f = pow(depth_given - depth_current, 2) / depth_variance;
  ir_new = lightingmodel->intensity(normal, pixel_to_camera(x, y, depth_current));
  shading_constraint_f = pow(ir_given - ir_new, 2) / ir_variance;

  shape_prior_f = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      shape_prior_f += dist(normals[i][j], normal); // TODO normal in the mid is counted twice
  shape_prior_f = w_d * shape_prior_f;

  // backward difference of depth
  depth_neighborhood[2][2] = depth_neighborhood[2][2] - h_depth;
  depth_current = tex2D(depth_current_tex, x, y) - h_depth;
  normal = normalmodel->normal(depth_neighborhood, make_int2(x, y));

  data_term_b = pow(depth_given - depth_current, 2) / depth_variance;
  ir_new = lightingmodel->intensity(normal, pixel_to_camera(x, y, depth_current));
  shading_constraint_b = pow(ir_given - ir_new, 2) / ir_variance;

  shape_prior_b = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      shape_prior_b += dist(normals[i][j], normal); // TODO normal in the mid is counted twice
  shape_prior_b = w_d * shape_prior_b;

  // d combined
  // FIXME look again at shading_constraint_f
  diff_d = ((data_term_f + shading_constraint_f + shape_prior_f) - (data_term_b + shading_constraint_b + shape_prior_b)) / h_depth;

  /* * * * */
  // Change in Material (changes shading constraint and material prior)
  /* * * * */

  // Get normals around the current point
  get_current_depth_neighborhood(make_int2(x, y), depth_neighborhood);
  normal = normalmodel->normal(depth_neighborhood, make_int2(x, y));

  float4 material_new = make_float4(0, 0, 0, 0);

  // forward difference of k_d
  material_new = tex2D(material_current_tex, x, y) + make_float4(0.5 * h_material, 0, 0, 0);
  lightingmodel->set_material_properties(material_new);
  ir_new = lightingmodel->intensity(normal, pixel_to_camera(x, y, tex2D(depth_sensor_tex, x, y)));
  shading_constraint_f = pow(ir_given - ir_new, 2) / ir_variance;

  material_prior_f = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      material_prior_f += len(tex2D(material_current_tex, x-1+i, y-1+j) - material_new);
  material_prior_f = w_m * material_prior_f;

  // backward difference of k_d
  material_new = tex2D(material_current_tex, x, y) - make_float4(0.5 * h_material, 0, 0, 0);
  lightingmodel->set_material_properties(material_new);
  ir_new = lightingmodel->intensity(normal, pixel_to_camera(x, y, tex2D(depth_sensor_tex, x, y)));
  shading_constraint_b = pow(ir_given - ir_new, 2) / ir_variance;

  material_prior_b = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      material_prior_b += len(tex2D(material_current_tex, x-1+i, y-1+j) - material_new);
  material_prior_b = w_m * material_prior_b;

  // k_d combined
  diff_kd = ((shading_constraint_f + material_prior_f) - (shading_constraint_b + material_prior_b)) / h_material;

  // forward difference of k_s
  material_new = tex2D(material_current_tex, x, y) + make_float4(0, 0.5 * h_material, 0, 0);
  lightingmodel->set_material_properties(material_new);
  ir_new = lightingmodel->intensity(normal, pixel_to_camera(x, y, tex2D(depth_sensor_tex, x, y)));
  shading_constraint_f = pow(ir_given - ir_new, 2) / ir_variance;

  material_prior_f = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      material_prior_f += len(tex2D(material_current_tex, x-1+i, y-1+j) - material_new);
  material_prior_f = w_m * material_prior_f;

  // backward difference of k_s
  material_new = tex2D(material_current_tex, x, y) - make_float4(0, 0.5 * h_material, 0, 0);
  lightingmodel->set_material_properties(material_new);
  ir_new = lightingmodel->intensity(normal, pixel_to_camera(x, y, tex2D(depth_sensor_tex, x, y)));
  shading_constraint_b = pow(ir_given - ir_new, 2) / ir_variance;

  material_prior_b = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      material_prior_b += len(tex2D(material_current_tex, x-1+i, y-1+j) - material_new);
  material_prior_b = w_m * material_prior_b;

  // k_s combined
  diff_ks = ((shading_constraint_f + material_prior_f) - (shading_constraint_b + material_prior_b)) / h_material;

  // forward difference of n
  material_new = tex2D(material_current_tex, x, y) + make_float4(0, 0, 0.5 * h_material, 0);
  lightingmodel->set_material_properties(material_new);
  ir_new = lightingmodel->intensity(normal, pixel_to_camera(x, y, tex2D(depth_sensor_tex, x, y)));
  shading_constraint_f = pow(ir_given - ir_new, 2) / ir_variance;

  material_prior_f = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      material_prior_f += len(tex2D(material_current_tex, x-1+i, y-1+j) - material_new);
  material_prior_f = w_m * material_prior_f;

  // backward difference of n
  material_new = tex2D(material_current_tex, x, y) - make_float4(0, 0, 0.5 * h_material, 0);
  lightingmodel->set_material_properties(material_new);
  ir_new = lightingmodel->intensity(normal, pixel_to_camera(x, y, tex2D(depth_sensor_tex, x, y)));
  shading_constraint_b = pow(ir_given - ir_new, 2) / ir_variance;

  material_prior_b = 0;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      material_prior_b += len(tex2D(material_current_tex, x-1+i, y-1+j) - material_new);
  material_prior_b = w_m * material_prior_b;

  // n combined
  diff_n = ((shading_constraint_f + material_prior_f) - (shading_constraint_b + material_prior_b)) / h_material;

  /* * * * */
  // Return values
  /* * * * */
  depth_out[index] = diff_d;
  material_out[index] = make_float4(diff_kd, diff_ks, diff_n, 0);

}

extern "C"
__global__ void energy(const int lightingmodel_enum, const int normalmodel_enum,
                        const float depth_variance, const float ir_variance,
                        const float w_d, const float w_m,
                       float *energy_data_term, float *energy_shading_constraint,
                       float *energy_shape_prior, float *energy_material_prior)
/*
  Calculates the energy per pixel.

  Returns four float arrays: Data, Shading Constraint, Shape prior, Material prior.

  Each array has to be summed along both axes, and then taken the sum.

  Input Textures:
    - depth_current_tex
    - depth_sensor_tex
    - ir_sensor_tex
    - material_current_tex
*/
{
  /* Steps to do:
    - Calculate ir intensity
    - Calculate normal of current point and adjacent pixels
    - Calculate energy: ( ir intensity - ir_sensor )
  */

  // TODO define variance as input parameters or outside in the numpy function?
  //const float depth_variance = 0.001;
  //const float ir_variance = 0.001;
  //const float w_d = 1;
  //const float w_m = 50;

  // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  PointLight light = PointLight(make_float3(0, 0, 0), 0.0);
  KinectCamera camera = KinectCamera(make_float3(0, 0, 0));

  LambertianLightingModel lambertianlightingmodel = LambertianLightingModel(light, camera);
  SpecularLightingModel specularlightingmodel = SpecularLightingModel(light, camera);
  LightingModel *lightingmodel;
  if (lightingmodel_enum == 0)
    lightingmodel = &lambertianlightingmodel;
  else if (lightingmodel_enum == 1)
    lightingmodel = &specularlightingmodel;
  lightingmodel->set_material_properties(tex2D(material_current_tex, x, y));

  NormalCalculator *normalmodel;
  NormalCross normalcross = NormalCross(camera);
  NormalPca normalpca = NormalPca(camera);
  if (normalmodel_enum == 0)
    normalmodel = &normalcross;
  else if (normalmodel_enum == 1)
    normalmodel = &normalpca;

  // Get normals around the current point
  float3 normals[3][3] = {make_float3(0, 0, 0)};
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      float depth_neighborhood[m_depth][m_depth];
      get_current_depth_neighborhood(make_int2(x - 1 + i, y - 1 + j), depth_neighborhood);
      normals[i][j] = normalmodel->normal(depth_neighborhood, make_int2(x - 1 + i, y - 1 + j));
    }
  }
  // The normal in the mid
  float3 normal = normals[1][1];

  // Data Term
  float depth_given = tex2D(depth_sensor_tex, x, y);
  float depth_current = tex2D(depth_current_tex, x, y);
  energy_data_term[index] = pow(depth_given - depth_current, 2) / depth_variance;

  // Shading Constraint
  float ir_given = tex2D(ir_sensor_tex, x, y);
  float ir_new = lightingmodel->intensity(normal, pixel_to_camera(camera, x, y, tex2D(depth_sensor_tex, x, y)));
  energy_shading_constraint[index] = pow(ir_given - ir_new, 2) / ir_variance;

  // Shape Prior
  float shape_prior = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      shape_prior += dist(normals[i][j], normal);
    }
  }
  energy_shape_prior[index] = w_d * shape_prior;

  // Material Prior
  float material_prior = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      material_prior += len(tex2D(material_current_tex, x-1+i, y-1+j) - tex2D(material_current_tex, x, y));
    }
  }
  energy_material_prior[index] = w_m * material_prior;

}

extern "C"
__global__ void intensity_image(int lightingmodel_enum, int normalmodel_enum, float *intensity_out)
/* Returns the infrared intensity image */
{
  // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  // Camera
  float3 camera_pos = make_float3(0, 0, 0);
  KinectCamera camera = KinectCamera(camera_pos);
  const float3 light_pos = make_float3(0, 0, 0);
  const float light_falloff = 0;
  PointLight light = PointLight(light_pos, light_falloff);

  NormalCalculator *normalmodel;
  NormalCross normalcross = NormalCross(camera);
  NormalPca normalpca = NormalPca(camera);
  if (normalmodel_enum == 0)
    normalmodel = &normalcross;
  else if (normalmodel_enum == 1)
    normalmodel = &normalpca;

  LambertianLightingModel lambertianlightingmodel = LambertianLightingModel(light, camera);
  SpecularLightingModel specularlightingmodel = SpecularLightingModel(light, camera);
  LightingModel *lightingmodel;
  if (lightingmodel_enum == 0)
    lightingmodel = & lambertianlightingmodel;
  else if (lightingmodel_enum == 1)
    lightingmodel = & specularlightingmodel;
  lightingmodel->set_material_properties(tex2D(material_current_tex, x, y));

  float depth_neighborhood[m_depth][m_depth];
  get_current_depth_neighborhood(make_int2(x, y), depth_neighborhood);
  float3 normal = normalmodel->normal(depth_neighborhood, make_int2(x, y));

  const float ir_new = lightingmodel->intensity(normal, pixel_to_camera(camera, x, y, tex2D(depth_sensor_tex, x, y)));

  intensity_out[index] = ir_new;
}


extern "C"
__global__ void normal_cross(float3 *normal_out)
/*
    Returns the normal image using Cross Product.
    Input Textures: depth_current_tex
*/
{
  // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  float3 camera_pos = make_float3(0, 0, 0);
  KinectCamera camera = KinectCamera(camera_pos);
  NormalCross normalcross = NormalCross(camera);

  float depth_neighborhood[m_depth][m_depth];
  get_current_depth_neighborhood(make_int2(x, y), depth_neighborhood);

  float3 normal = normalcross.normal(depth_neighborhood, make_int2(x, y));
  normal_out[index] = normal_colorize(normal);
}

extern "C"
__global__ void normal_pca(float3 *normal_out)
/*
    Returns the normal image using PCA
    Input Textures: depth_current_tex
*/
{
  // Indexing
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int elementPitch = blockDim.x * gridDim.x;
  const int index = y * elementPitch + x;

  float3 camera_pos = make_float3(0, 0, 0);
  KinectCamera camera = KinectCamera(camera_pos);
  NormalPca normalpca = NormalPca(camera);

  float depth_neighborhood[m_depth][m_depth];
  get_current_depth_neighborhood(make_int2(x, y), depth_neighborhood);

  float3 normal = normalpca.normal(depth_neighborhood, make_int2(x, y));
  normal_out[index] = normal_colorize(normal);
}

#endif