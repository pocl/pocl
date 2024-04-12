/*==============================================================
 * Copyright © 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 * ============================================================= */

/* Heterogeneous distributed Jacobian computation sample using SYCL and PoCL-R.
 */
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

struct subarray {
  int device_i;
  int x_size, y_size; /* Subarray size excluding border rows and columns */
  int l_nbh_offt;     /* Offset predecessor data to update */
};

size_t device_count = 0;

#define ROW_SIZE(S) ((S).x_size + 2)
#define XY_2_IDX(X,Y,S) (((Y)+1)*ROW_SIZE(S)+((X)+1))

/* Subroutine to create and initialize initial state of input subarrays */
void InitDeviceArrays(std::vector<double*>& bufs, sycl::queue& q, struct subarray *sub)
{
    size_t total_size = (sub->x_size + 2) * (sub->y_size + 2);

    double *A = sycl::malloc_host < double >(total_size, q);
    assert (A != nullptr);

    double *A_dev_1 = sycl::malloc_device < double >(total_size, q);
    assert (A_dev_1 != nullptr);

    double *A_dev_2 = sycl::malloc_device < double >(total_size, q);
    assert (A_dev_2 != nullptr);

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
    q.memcpy(A_dev_1, A, sizeof(double) * total_size);
    q.memcpy(A_dev_2, A, sizeof(double) * total_size);

    q.wait();
}

int main(int argc, char *argv[])
{
    if (sycl::platform::get_platforms().size() == 0) {
      std::cerr << "No SYCL platforms found.\n";
      return EXIT_FAILURE;
    }
    auto platform = sycl::platform::get_platforms().at(0);
    auto devices = platform.get_devices();
    std::cout << "Running on " << devices.size() << " device(s)\n";

    std::vector<double*> A_device(devices.size() * 2);
    std::vector<subarray> subarrays(devices.size());
    std::vector<sycl::queue*> queues(devices.size());

    std::barrier host_sync_point(devices.size());

    device_count = devices.size();

    // Compute the work shares and initialize data.
    for (size_t device_id = 0; device_id < device_count; ++device_id) {
      auto device = devices.at(device_id);
      std::cout << "\tUsing device: "
                << device.get_info<sycl::info::device::name>()
                << std::endl;

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

      sycl::queue * q = new sycl::queue(device);
      queues[device_id] = q;
      InitDeviceArrays(A_device, *q, &sub);
    }

    std::atomic<double> norm = 0.0;
    double final_norm = 0.0;

    std::for_each(
        std::execution::par, std::begin(subarrays), std::end(subarrays),
        [&](subarray &my_subarray) {
          auto device = devices.at(my_subarray.device_i);

          /* Initialization of runtime and initial state of data */
          auto &q = *queues.at(my_subarray.device_i);

          host_sync_point.arrive_and_wait();

          for (int i = 0; i < Niter; ++i) {
            double *a = A_device[my_subarray.device_i * 2 + i % 2];
            double *a_out = A_device[my_subarray.device_i * 2 + (i + 1) % 2];
            {
              /* Calculate values on borders to initiate communications early */
              q.submit([&](auto &h) {
                 sycl::stream out(1024, 256, h);
                 h.parallel_for(
                     sycl::range(my_subarray.x_size), [=](auto index) {
                       int column = index[0];
                       int idx = XY_2_IDX(column, 0, my_subarray);
                       a_out[idx] = 0.25 * (a[idx - 1] + a[idx + 1] +
                                            a[idx - ROW_SIZE(my_subarray)] +
                                            a[idx + ROW_SIZE(my_subarray)]);

                       idx = XY_2_IDX(column, my_subarray.y_size - 1,
                                      my_subarray);
                       a_out[idx] = 0.25 * (a[idx - 1] + a[idx + 1] +
                                            a[idx - ROW_SIZE(my_subarray)] +
                                            a[idx + ROW_SIZE(my_subarray)]);
                     });
               }).wait();
            }

            host_sync_point.arrive_and_wait();

            /* Perform 1D halo-exchange with neighbours */
            if (my_subarray.device_i != 0) {
              int idx = XY_2_IDX(0, 0, my_subarray);
              double *cwin =
                  A_device[(my_subarray.device_i - 1) * 2 + (i + 1) % 2];
              q.memcpy(&cwin[my_subarray.l_nbh_offt], &a_out[idx],
                       my_subarray.x_size * sizeof(double));
              q.wait();
            }

            host_sync_point.arrive_and_wait();

            if (my_subarray.device_i != (device_count - 1)) {
              int idx = XY_2_IDX(0, my_subarray.y_size - 1, my_subarray);
              double *cwin =
                  A_device[(my_subarray.device_i + 1) * 2 + (i + 1) % 2];
              q.memcpy(&cwin[1], &a_out[idx],
                       my_subarray.x_size * sizeof(double));
              q.wait();
            }

            host_sync_point.arrive_and_wait();

            {
              q.submit([&](auto &h) {
                 h.parallel_for(
                     sycl::range(my_subarray.x_size, my_subarray.y_size - 2),
                     [=](auto index) {
                       int idx = XY_2_IDX(index[0], index[1] + 1, my_subarray);
                       a_out[idx] = 0.25 * (a[idx - 1] + a[idx + 1] +
                                            a[idx - ROW_SIZE(my_subarray)] +
                                            a[idx + ROW_SIZE(my_subarray)]);
                     });
               }).wait();
            }

            host_sync_point.arrive_and_wait();

            /* Calculate and report norm value after given number of iterations
             */
            if ((NormIteration > 0) &&
                ((NormIteration - 1) == i % NormIteration)) {
              double device_norm = 0.0;
              {
                sycl::buffer<double> norm_buf(&device_norm, 1);
                q.submit([&](auto &h) {
                   auto sumr = sycl::reduction(norm_buf, h, sycl::plus<>());
                   h.parallel_for(
                       sycl::range(my_subarray.x_size, my_subarray.y_size),
                       sumr, [=](auto index, auto &v) {
                         int idx = XY_2_IDX(index[0], index[1], my_subarray);
                         double diff = a_out[idx] - a[idx];
                         v += (diff * diff);
                       });
                 }).wait();
              }

              /* Get global norm value */
              host_sync_point.arrive_and_wait();
              norm += device_norm;

              host_sync_point.arrive_and_wait();

              if (my_subarray.device_i == 0) {
                final_norm = sqrt(norm);
                norm = 0.0;
              }
            }

            host_sync_point.arrive_and_wait();
            /* Ensure all communications complete before next iteration */
            q.wait();
            host_sync_point.arrive_and_wait();
          }
        });

    // For Nx 1024 the norm at 100 should be 2.282558.
    if (fabs(final_norm - 2.282558) < 0.001)
      printf("OK\n");
    else
      printf("FAIL! Delta == %f\n", fabs(final_norm - 2.282558));
    return 0;
}
