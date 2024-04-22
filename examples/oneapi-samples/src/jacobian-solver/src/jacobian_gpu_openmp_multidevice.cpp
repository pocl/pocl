/*==============================================================
 * Copyright © 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ============================================================= */

/* Heterogeneous distributed Jacobian computation sample using OpenMP and PoCL-R.
 */

#include <omp.h>
#include <atomic>
#include <barrier>
#include <execution>
#include <iostream>
#include <sycl.hpp>
#include <vector>

//const int Nx = 16384; /* Grid size */
const int Nx = 1024;
const int Ny = Nx;
const int Niter = 100; /* Nuber of algorithm iterations */
const int NormIteration = 10; /* Recaluculate norm after given number of iterations. 0 to disable norm calculation */

size_t device_count = 0;

struct subarray {
  int device_i;       /* Device's index [0..device_count) */
  int x_size, y_size; /* Subarray size excluding border rows and columns */
  int l_nbh_offt;     /* Offset predecessor data to update */
};

#define ROW_SIZE(S) ((S).x_size + 2)
#define XY_2_IDX(X,Y,S) (((Y)+1)*ROW_SIZE(S)+((X)+1))

/* Subroutine to create and initialize initial state of input subarrays */
void InitDeviceArrays(double** bufs, struct subarray *sub)
{
  size_t total_size = (sub->x_size + 2) * (sub->y_size + 2);

  int host_id = omp_get_initial_device();
  double *A = (double*) malloc(total_size * sizeof(double));

  double *A_dev_1 = (double*) omp_target_alloc_device(total_size * sizeof(double), sub->device_i);
  double *A_dev_2 = (double*) omp_target_alloc_device(total_size * sizeof(double), sub->device_i);

  bufs[sub->device_i * 2] = A_dev_1;
  bufs[sub->device_i * 2 + 1] = A_dev_2;

  for (int i = 0; i < (sub->y_size + 2); i++)
    for (int j = 0; j < (sub->x_size + 2); j++)
      A[i * (sub->x_size + 2) + j] = 0.0;

  if (sub->device_i == 0) /* set top boundary */
    for (int i = 1; i <= sub->x_size; i++)
      A[i] = 1.0; /* set bottom boundary */
  if (sub->device_i == (device_count - 1))
    for (int i = 1; i <= sub->x_size; i++)
      A[(sub->x_size + 2) * (sub->y_size + 1) + i] = 10.0;

  for (int i = 1; i <= sub->y_size; i++) {
    int row_offt = i * (sub->x_size + 2);
    A[row_offt] = 1.0;      /* set left boundary */
    A[row_offt + sub->x_size + 1] = 1.0;    /* set right boundary */
  }

  /* Move input arrays to device */
  omp_target_memcpy(A_dev_1, A, sizeof(double) * total_size, 0, 0, sub->device_i, host_id);
  omp_target_memcpy(A_dev_2, A, sizeof(double) * total_size, 0, 0, sub->device_i, host_id);
  free(A);
}

int main(int argc, char *argv[])
{
  device_count = omp_get_num_devices();
  if (device_count == 0) {
    std::cerr << "No OpenMP target devices found.\n";
    return EXIT_FAILURE;
  }
  std::cout << "Running on " << device_count << " device(s)\n";
  std::cout << "Initial (host) device: " << omp_get_initial_device() << "\n";

  double* A_device[device_count * 2];
  std::vector<subarray> subarrays(device_count);

  // Compute the work shares and initialize data.
  for (size_t device_id = 0; device_id < device_count; ++device_id) {
    std::cout << "Initializing device " << device_id << std::endl;

    // Setup subarray size and layout processed by a device.
    struct subarray& sub = subarrays.at(device_id);

    sub.device_i = device_id;
    sub.y_size = Ny / device_count;
    sub.x_size = Nx;
    sub.l_nbh_offt = (sub.x_size + 2) * (sub.y_size + 1) + 1;

    int tail = sub.y_size % device_count;
    if (tail != 0) {
      if (sub.device_i < tail)
        sub.y_size++;
      if ((sub.device_i > 0) && ((sub.device_i - 1) < tail))
        sub.l_nbh_offt += (sub.x_size + 2);
    }

    InitDeviceArrays(A_device, &sub);
  }

  std::atomic<double> norm = 0.0;
  double final_norm = 0.0;

  // Spawn parallel threads explicitly since we cannot use barriers
  // inside "omp parallel for" segments (!).
  omp_set_num_threads(device_count);
#pragma omp parallel
    {
      int device_i = omp_get_thread_num();
      std::cerr << "Running " << device_i << " device thread\n";
      auto my_subarray = subarrays.at(device_i);

#pragma omp target data map(to: Niter, my_subarray, A_device, NormIteration) device(device_i)
      {
      for (int i = 0; i < Niter; ++i) {
        double *a = A_device[device_i * 2 + i % 2];
        double *a_out = A_device[device_i * 2 + (i + 1) % 2];
#pragma omp target data map(to: a, a_out) device(device_i)
        {
          /* Offload compute loop to the device */
#pragma omp target teams distribute parallel for is_device_ptr(a, a_out) device(device_i)
          /* Calculate values on borders to initiate communications early */
          for (int _column = 0; _column < my_subarray.x_size; ++_column) {
            int idx = XY_2_IDX(_column, 0, my_subarray);
            a_out[idx] = 0.25 * (a[idx - 1] + a[idx + 1] + a[idx - ROW_SIZE(my_subarray)] +
                                 a[idx + ROW_SIZE(my_subarray)]);

            idx = XY_2_IDX(_column, my_subarray.y_size - 1, my_subarray);
            a_out[idx] = 0.25 * (a[idx - 1] + a[idx + 1] + a[idx - ROW_SIZE(my_subarray)] +
                                 a[idx + ROW_SIZE(my_subarray)]);
          }
        }
#pragma omp barrier

        /* Perform 1D halo-exchange with neighbours */
        if (device_i != 0) {
            int idx = XY_2_IDX(0, 0, my_subarray);
            double *cwin = A_device[(device_i - 1) * 2 + (i + 1) % 2];
            omp_target_memcpy(&cwin[my_subarray.l_nbh_offt], &a_out[idx],
                              my_subarray.x_size * sizeof (double), 0, 0,
                              device_i - 1, device_i);
        }

#pragma omp barrier

        if (my_subarray.device_i != (device_count - 1)) {
            int idx = XY_2_IDX(0, my_subarray.y_size - 1, my_subarray);
            double *cwin = A_device[(device_i + 1) * 2  + (i + 1) % 2];
            omp_target_memcpy(&cwin[1], &a_out[idx],
                              my_subarray.x_size * sizeof (double), 0, 0,
                              device_i + 1, device_i);
        }

#pragma omp barrier

        /* Offload compute loop to the device */
#pragma omp target teams distribute parallel for is_device_ptr(a, a_out) collapse(2) device(device_i)
        /* Recalculate internal points in parallel with communication */
        for (int row = 1; row < my_subarray.y_size - 1; ++row) {
          for (int column = 0; column < my_subarray.x_size; ++column) {
            int idx = XY_2_IDX(column, row, my_subarray);
            a_out[idx] = 0.25 * (a[idx - 1] + a[idx + 1] + a[idx - ROW_SIZE(my_subarray)]
                                 + a[idx + ROW_SIZE(my_subarray)]);
          }
        }

#pragma omp barrier

        /* Calculate and report norm value after given number of iterations */
        if ((NormIteration > 0) && ((NormIteration - 1) == i % NormIteration)) {
            double device_norm = 0.0;

            /* Offload compute loop to the device */
            /* This fails on PoCL-CPU (race condition on a global value?) */
#pragma omp target teams distribute parallel for is_device_ptr(a, a_out) reduction(+:device_norm) collapse(2) device(device_i)
            /* Calculate and report norm value after given number of iterations */
            for (int row = 0; row < my_subarray.y_size; ++row) {
              for (int column = 0; column < my_subarray.x_size; ++column) {
                int idx = XY_2_IDX(column, row, my_subarray);
                double diff = a_out[idx] - a[idx];
                device_norm += diff*diff;
              }
            }

            /* Update global norm value atomically. TODO: OMP atomic*/
#pragma omp barrier
            norm += device_norm;
#pragma omp barrier

            if (my_subarray.device_i == 0) {
                final_norm = sqrt(norm);
                norm = 0.0;
            }
        }
#pragma omp barrier
      }
      }
    }
  // For Nx 1024 the norm at 100 should be 2.282558.
  if (fabs(final_norm - 2.282558) < 0.001)
    printf("OK\n");
  else
    printf("FAIL! Delta == %f\n", fabs(final_norm - 2.282558));

  return 0;
}
