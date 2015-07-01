// ----------------------------------------------------------------------------
// Numerical diagonalization of 3x3 matrcies
// Copyright (C) 2006  Joachim Kopp
// http://www.mpi-hd.mpg.de/personalhomes/globes/3x3/
//
// Johannes Mikulasch: Small ammendments for use in cuda, added eigen_sort
//
// ----------------------------------------------------------------------------
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
// ----------------------------------------------------------------------------

#ifndef EIGEN_CU
#define EIGEN_CU

/*------------------------------------*/
/*---------- DECLARATIONS ------------*/
/*------------------------------------*/

extern "C" __device__ int eigen_symmetric3x3(double A[3][3], double Q[3][3], double w[3]);
/*  Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
    matrix A using the Jacobi algorithm.
    The upper triangular part of A is destroyed during the calculation,
    the diagonal elements are read but not destroyed, and the lower
    triangular elements are not referenced at all.
    Parameters:
     A: The symmetric input matrix
     Q: Storage buffer for eigenvectors
     w: Storage buffer for eigenvalues
    Return value:
     0: Success
    -1: Error (no convergence)
    License: GNU3
    by Joachim Kopp
    http://www.mpi-hd.mpg.de/personalhomes/globes/3x3/ */

extern "C" __device__ void eigen_sort(float3 eigvectors[3], double eigvalues[3]);
/*  Sorts the eigenvectors and eigenvalues in-place, beginning with the smallest eigenvalue */


/*------------------------------------*/
/*---------- DEFINITIONS -------------*/
/*------------------------------------*/

#define SQR(x)      ((x)*(x))

extern "C" __device__ int eigen_symmetric3x3(double A[3][3], double Q[3][3], double w[3])
{
  const int n = 3;
  float sd, so;                  // Sums of diagonal resp. off-diagonal elements
  float s, c, t;                 // sin(phi), cos(phi), tan(phi) and temporary storage
  float g, h, z, theta;          // More temporary storage
  float thresh;

  // Initialize Q to the identitity matrix
  for (int i=0; i < n; i++)
  {
    Q[i][i] = 1.0;
    for (int j=0; j < i; j++)
      Q[i][j] = Q[j][i] = 0.0;
  }

  // Initialize w to diag(A)
  for (int i=0; i < n; i++)
    w[i] = A[i][i];

  // Calculate SQR(tr(A))
  sd = 0.0;
  for (int i=0; i < n; i++)
    sd += fabs(w[i]);
  sd = SQR(sd);

  // Main iteration loop
  for (int nIter=0; nIter < 50; nIter++)
  {
    // Test for convergence
    so = 0.0;
    for (int p=0; p < n; p++)
      for (int q=p+1; q < n; q++)
        so += fabs(A[p][q]);
    if (so == 0.0)
      return 0;

    if (nIter < 4)
      thresh = 0.2 * so / SQR(n);
    else
      thresh = 0.0;

    // Do sweep
    for (int p=0; p < n; p++)
      for (int q=p+1; q < n; q++)
      {
        g = 100.0 * fabs(A[p][q]);
        if (nIter > 4  &&  fabs(w[p]) + g == fabs(w[p])
                       &&  fabs(w[q]) + g == fabs(w[q]))
        {
          A[p][q] = 0.0;
        }
        else if (fabs(A[p][q]) > thresh)
        {
          // Calculate Jacobi transformation
          h = w[q] - w[p];
          if (fabs(h) + g == fabs(h))
          {
            t = A[p][q] / h;
          }
          else
          {
            theta = 0.5 * h / A[p][q];
            if (theta < 0.0)
              t = -1.0 / (sqrt(1.0 + SQR(theta)) - theta);
            else
              t = 1.0 / (sqrt(1.0 + SQR(theta)) + theta);
          }
          c = 1.0/sqrt(1.0 + SQR(t));
          s = t * c;
          z = t * A[p][q];

          // Apply Jacobi transformation
          A[p][q] = 0.0;
          w[p] -= z;
          w[q] += z;
          for (int r=0; r < p; r++)
          {
            t = A[r][p];
            A[r][p] = c*t - s*A[r][q];
            A[r][q] = s*t + c*A[r][q];
          }
          for (int r=p+1; r < q; r++)
          {
            t = A[p][r];
            A[p][r] = c*t - s*A[r][q];
            A[r][q] = s*t + c*A[r][q];
          }
          for (int r=q+1; r < n; r++)
          {
            t = A[p][r];
            A[p][r] = c*t - s*A[q][r];
            A[q][r] = s*t + c*A[q][r];
          }

          // Update eigenvectors
          for (int r=0; r < n; r++)
          {
            t = Q[r][p];
            Q[r][p] = c*t - s*Q[r][q];
            Q[r][q] = s*t + c*Q[r][q];
          }
        }
      }
  }
  return -1;
}

extern "C"
__device__ void eigen_sort3(float3 eigvectors[3], double eigvalues[3])
{

  // Copy temporary values for swapping later
  float3 tmp_vectors[3];
  float tmp_values[3];
  for (int i = 0; i < 3; ++i) {
    tmp_vectors[i] = eigvectors[i];
    tmp_values[i] = eigvalues[i];
  }

  int index_0, index_1, index_2;
  if (eigvalues[0] <= eigvalues[1] && eigvalues[1] <= eigvalues[2]) {
    index_0 = 0; index_1 = 1; index_2 = 2;
  } else if (eigvalues[0] <= eigvalues[2] && eigvalues[2] <= eigvalues[1]) {
    index_0 = 0; index_1 = 2; index_2 = 1;
  } else if (eigvalues[1] <= eigvalues[0] && eigvalues[0] <= eigvalues[2]) {
    index_0 = 1; index_1 = 0; index_2 = 2;
  } else if (eigvalues[1] <= eigvalues[2] && eigvalues[2] <= eigvalues[0]) {
    index_0 = 1; index_1 = 2; index_2 = 0;
  } else if (eigvalues[2] <= eigvalues[0] && eigvalues[0] <= eigvalues[1]) {
    index_0 = 2; index_1 = 0; index_2 = 1;
  } else {
    index_0 = 2; index_1 = 1; index_2 = 0;
  }

  eigvalues[0] = tmp_values[index_0];
  eigvalues[1] = tmp_values[index_1];
  eigvalues[2] = tmp_values[index_2];
  eigvectors[0] = tmp_vectors[index_0];
  eigvectors[1] = tmp_vectors[index_1];
  eigvectors[2] = tmp_vectors[index_2];
}

#endif