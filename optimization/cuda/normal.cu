// Author: Johannes Mikulasch
// July 2015

#ifndef NORMAL_CU
#define NORMAL_CU

#ifndef m_depth
  #define m_depth 5
#endif

#include "utils.cu"
#include "eigen.cu"
#include "cameras.cu"

class NormalCalculator {
public:
  virtual __device__ float3 normal(const float depth_neighborhood[m_depth][m_depth], const int2 pos) = 0;
};

class NormalCross : public NormalCalculator {

public:
  __device__ NormalCross() {}
  __device__ NormalCross(Camera &camera)
    : m_camera(&camera) { }
  __device__ float3 normal(const float depth_neighborhood[m_depth][m_depth], const int2 pos)
  {

    const int midpoint = (m_depth - 1) / 2.;

    if (depth_neighborhood[midpoint][midpoint] == 0)
      return make_float3(0, 0, 0);

    // Find the neighbors of the neighbor in camera coords
    const float3 point_b = pixel_to_camera(*m_camera, pos.x, pos.y-1, depth_neighborhood[midpoint - 1][midpoint]);
    const float3 point_d = pixel_to_camera(*m_camera, pos.x-1, pos.y, depth_neighborhood[midpoint][midpoint - 1]);
    const float3 point_f = pixel_to_camera(*m_camera, pos.x+1, pos.y, depth_neighborhood[midpoint][midpoint + 1]);
    const float3 point_h = pixel_to_camera(*m_camera, pos.x, pos.y+1, depth_neighborhood[midpoint + 1][midpoint]);

    // Calculate Normal
    const float3 vectorDF = point_f - point_d;
    const float3 vectorHB = point_b - point_h;
    const float3 normal = normalize(cross(vectorDF, vectorHB));
    return normal;
  }

private:
  Camera *m_camera;
};

class NormalPca : public NormalCalculator {

public:
  __device__ NormalPca() {}
  __device__ NormalPca(Camera &camera)
   : m_camera(&camera) { }
  __device__ float3 normal(const float depth_neighborhood[m_depth][m_depth], const int2 pos)
  {
    /*
      Calculate the normal in the midpoint (2, 2) in a 5x5 neighborhood using PCA
      pos.x, pos.y: pixel coordinates of the midpoint
      Idea from: http://pointclouds.org/documentation/tutorials/normal_estimation.php
    */

    const int points_n = m_depth * m_depth;

    float3 world_points[points_n];
    for (int i = 0; i < m_depth; ++i)
      for (int j = 0; j < m_depth; ++j)
        world_points[j*m_depth + i] = pixel_to_camera(*m_camera, pos.x - 2 + i, pos.y - 2 + j, depth_neighborhood[j][i]);

    float3 center = world_points[(points_n - 1) / 2];

    if (center.z == 0)
       return make_float3(0, 0, 0);

    // Search radius depends on Z
    //const float radius = 15. / 368. * -center.z;

    // Search radius depends on the mean distance of all points
    float avg_dist = 0;
    for (int i = 0; i < points_n; ++i)
      avg_dist += dist(world_points[i], center);
    avg_dist /= points_n;
    const float radius = avg_dist * 2;
    //const float radius = 200;

    // Calculate the average point location
    float3 avg = {0};
    int count = 0;
    for (int i = 0; i < points_n; ++i) {
      if (dist(world_points[i], center) <= radius && world_points[i].z != 0) {
         avg = avg + world_points[i];
         ++count;
       }
    }
    avg = avg / count;

    // Calculate the covariance matrix
    double cov[3][3] = {0};
    for (int i = 0; i < points_n; ++i) {
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
private:
  Camera *m_camera;
};

#endif
