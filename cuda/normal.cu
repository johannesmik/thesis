// Author: Johannes Mikulasch
// July 2015

#ifndef NORMAL_CU
#define NORMAL_CU

#include "utils.cu"
#include "eigen.cu"

__device__ float3 normal_cross(const float depth_neighborhood[25], const int x_n, const int y_n)
{
  /*
     Calculates the normals in the midpoint (2, 2) in a 5x5 neighborhood using a cross product
     x_n, y_n: pixel coordinates of the midpoint
  */

  if (depth_neighborhood[12] == 0)
    return make_float3(0, 0, 0);

  // Find the neighbors of the neighbor in camera coords
  const float3 point_b = pixel_to_camera(x_n, y_n-1, depth_neighborhood[7]);
  const float3 point_d = pixel_to_camera(x_n-1, y_n, depth_neighborhood[11]);
  const float3 point_f = pixel_to_camera(x_n+1, y_n, depth_neighborhood[13]);
  const float3 point_h = pixel_to_camera(x_n, y_n+1, depth_neighborhood[17]);

  // Calculate Normal
  const float3 vectorDF = point_f - point_d;
  const float3 vectorHB = point_b - point_h;
  const float3 normal = normalize(cross(vectorDF, vectorHB));
  return normal;
}

__device__ float3 normal_pca(const float depth_neighborhood[25], const int x_n, const int y_n)
{
  /*
    Calculate the normal in the midpoint (2, 2) in a 5x5 neighborhood using PCA
    x_n, y_n: pixel coordinates of the midpoint
    Idea from: http://pointclouds.org/documentation/tutorials/normal_estimation.php
  */

  float3 world_points[25];
  for (int i = 0; i < 5; ++i)
    for (int j = 0; j < 5; ++j)
      world_points[j*5 + i] = pixel_to_camera(x_n - 2 + i, y_n - 2 + j, depth_neighborhood[j*5 + i]);

  float3 center = world_points[12];

  if (center.z == 0)
     return make_float3(0, 0, 0);

  // Search radius depends on Z
  //const float radius = 15. / 368. * -center.z;

  // Search radius depends on the mean distance of all points
  float avg_dist = 0;
  for (int i = 0; i < 25; ++i)
    avg_dist += dist(world_points[i], center);
  avg_dist /= 25;
  const float radius = avg_dist;

  // Calculate the average point location
  float3 avg = {0};
  int count = 0;
  for (int i = 0; i < 25; ++i) {
    if (dist(world_points[i], center) <= radius && world_points[i].z != 0) {
       avg = avg + world_points[i];
       ++count;
     }
  }
  avg = avg / count;

  // Calculate the covariance matrix
  double cov[3][3] = {0};
  for (int i = 0; i < 25; ++i) {
    if (dist(world_points[i], center) <= radius && world_points[i].z != 0) {
      // XX, XY, XZ, YY, YZ, ZZ
      cov[0][0] += (double) (world_points[i].x - avg.x) * (world_points[i].x - avg.x) / count;
      cov[0][1] += (double) (world_points[i].x - avg.x) * (world_points[i].y - avg.y) / count;
      cov[0][2] += (double) (world_points[i].x - avg.x) * (world_points[i].z - avg.z) / count;
      cov[1][1] += (double) (world_points[i].y - avg.y) * (world_points[i].y - avg.y) / count;
      cov[1][2] += (double) (world_points[i].y - avg.y) * (world_points[i].z - avg.z) / count;
      cov[2][2] += (double) (world_points[i].z - avg.z) * (world_points[i].z - avg.z) / count;
    }
  }

  // Supplement the symmetric values (not necessary needed)
  //cov[1][0] = cov[0][1]; cov[2][0] = cov[0][2]; cov[2][1] = cov[1][2];

  // Calculate the eigenvectors and eigenvalues and sort them
  double v_t[3][3];
  float3 eigenvectors[3] = {0};
  double eigenvalues[3] = {0};
  eigen_symmetric3x3(cov, v_t, eigenvalues);
  for (int i = 0; i < 3; ++i)
    eigenvectors[i] = make_float3(v_t[0][i], v_t[1][i], v_t[2][i]);
  eigen_sort3(eigenvectors, eigenvalues);

  // Take the smallest eigenvector and reorient it to the camera (0, 0, 0)
  if (dot(eigenvectors[0], center) > 0)
    return -eigenvectors[0];
  else
    return eigenvectors[0];
}

#endif
