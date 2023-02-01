/* pocl_local_size.c - Different means for optimizing the local size.

   Copyright (c) 2011-2019 pocl developers
                 2020 Pekka Jääskeläinen

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#include "pocl_local_size.h"

/* Euclid's algorithm for the Greatest Common Divisor */
static inline size_t
gcd (size_t a, size_t b)
{
  int c;
  while (a)
    {
      c = a;
      a = b % a;
      b = c;
    }
  return b;
}

/* Find the largest divisor of dividend which is less than limit */
static inline size_t
upper_divisor (size_t dividend, size_t limit)
{
  /* The algorithm is currently not very smart, we
   * start from limit and subtract until we find something
   * that divides dividend. In optimal conditions this is found
   * quickly, but it takes limit steps if dividend is prime.
   * TODO FIXME improve algorithm
   */
  if (dividend < limit)
    return dividend;  // small optimization
  assert (limit > 0); // should never be called with limit == 0
  while (dividend % limit != 0)
    --limit;
  return limit;
}

/* Check that a divides b and b divides c */
static inline int
divide_chain (size_t a, size_t b, size_t c)
{
  return (b % a == 0 && c % b == 0);
}

void
pocl_default_local_size_optimizer (cl_device_id dev, cl_kernel kernel,
                                   unsigned device_i,
                                   size_t global_x, size_t global_y,
                                   size_t global_z, size_t *local_x,
                                   size_t *local_y, size_t *local_z)
{
  /* Tries figure out a local size which utilizes all the device's resources
   * efficiently. Assume work-groups are scheduled to compute units, so
   * try to split it to a number of work groups at the equal to the number
   * of CUs, while still trying to respect the preferred WG size multiple
   * (for better SIMD instruction utilization).
   */
  size_t max_local_x, max_local_y, max_local_z;

  max_local_x = min (dev->max_work_item_sizes[0], global_x);
  max_local_y = min (dev->max_work_item_sizes[1], global_y);
  max_local_z = min (dev->max_work_item_sizes[2], global_z);

  size_t preferred_wg_multiple = dev->preferred_wg_size_multiple;
  size_t max_group_size = dev->max_work_group_size;

  if (!preferred_wg_multiple) /* unlikely */
    preferred_wg_multiple = 1;

  POCL_MSG_PRINT_INFO ("Preferred WG size multiple %zu\n",
                       preferred_wg_multiple);

  /* However, we have some constraints about the local size:
   * 1. local_{x,y,z} must divide global_{x,y,z} exactly, at least
   *    as long as we only support uniform group sizes (i.e. OpenCL 1.x);
   * 2. each of local_{x,y,z} must be less than the corresponding max size
   *    for the device;
   * 3. the product of local_{x,y,z} must be less than the maximum local
   *    work-group size.
   *
   * Due to constraint 1., we may not have the possibility to proceed by
   * multiples of the preferred_wg_multiple (e.g. if preferred = 16 and
   * global size = 24). Our stepping granularity in each direction will
   * therefore be the GCD of the global size in that direction and the
   * preferred wg size.
   *
   * Note that the grain might actually be as low as 1, if the two values
   * are coprimes (e.g. preferred = 8, global size = 17). There is no good
   * solution in this case, and there's nothing we can do about it. On the
   * opposite side of the spectrum, we might be lucky and grain_* =
   * preferred_wg_multiple (this is the case e.g. if the programmer already
   * checked for the preferred wg multiple and rounded the global size up
   * to the multiple of it).
   */

  const size_t grain_x = gcd (preferred_wg_multiple, global_x);
  const size_t grain_y = gcd (preferred_wg_multiple, global_y);
  const size_t grain_z = gcd (preferred_wg_multiple, global_z);

  /* We now want to get the largest multiple of the grain size that still
   * divides global_* _and_ is less than the maximum local size in each
   * direction.
   *
   * So we have G = K*g and we want to find k such that k*g < M and
   * k*g still divides G, i.e. k must divide K.
   * The largest multiple of g that is less than M can be found as
   * (M/g)*g (integer division), so our upper bound for k is k' = M/g.
   */

  /*                      /------- K ------\  /-------- k' -------\  */
  *local_x = upper_divisor (global_x / grain_x, max_local_x / grain_x);
  *local_x *= grain_x;

  if (max_local_y > 1)
    {
      *local_y = upper_divisor (global_y / grain_y, max_local_y / grain_y);
      *local_y *= grain_y;
    }
  else
    *local_y = 1;

  if (max_local_z > 1)
    {
      *local_z = upper_divisor (global_z / grain_z, max_local_z / grain_z);
      *local_z *= grain_z;
    }
  else
    *local_z = 1;

  /* So we now have the largest possible local sizes that divide the global
   * sizes while being multiples of the grain size.
   * We still have to ensure that the work-group size overall is not larger
   * than the maximum allowed, and we have to do this while preserving the
   * 'local divides global' condition, and we would like to preserve the
   * 'multiple of grain' too, if possible.
   * We always reduce z first, then y, then x, on the assumption that
   * kernels will work with x varying faster, and thus being a better
   * vectorization candidate, followed by y and then by z. (This assumption
   * is in some sense sanctioned by the standard itself, see e.g. the
   * get_{global,local}_linear_id functions in OpenCL 2.x)
   * TODO this might not be optimal in all cases. For example, devices with
   * a hardware sampler might benefit from more evenly sized work-groups
   * for kernels that use images. Some kind of kernel + device analysis
   * would be needed here.
   */

  while (*local_x * *local_y * *local_z > max_group_size)
    {
      /* We are going to try three strategies, in order:
       *
       * Halving a coordinate, if the halved coordinate is still a multiple
       * of the grain size and a divisor of the global size.
       *
       * Setting the coordinates with the smallest grain to 1,
       * since they aren't good candidates for vectorizations anyway.
       *
       * Setting to 1 any coordinate, as a desperate measure.
       */

#define TRY_HALVE(coord)                                                      \
  if ((*local_##coord & 1) == 0                                               \
      && divide_chain (grain_##coord, *local_##coord / 2, global_##coord))    \
    {                                                                         \
      *local_##coord /= 2;                                                    \
      continue;                                                               \
    }

#define TRY_LEAST_GRAIN(c1, c2, c3)                                           \
  if (*local_##c1 > 1 && grain_##c1 <= grain_##c2                             \
      && grain_##c1 <= grain_##c3)                                            \
    {                                                                         \
      *local_##c1 = 1;                                                        \
      continue;                                                               \
    }

#define DESPERATE_CASE(coord)                                                 \
  if (*local_##coord > 1)                                                     \
    {                                                                         \
      *local_##coord = 1;                                                     \
      continue;                                                               \
    }
      /* Halving attempt first */
      TRY_HALVE (z) else TRY_HALVE (y) else TRY_HALVE (x);

      /* Ok no luck. Find the coordinate with the smallest grain and
       * kill that */
      TRY_LEAST_GRAIN (z, x, y)
      else TRY_LEAST_GRAIN (y, z, x) else TRY_LEAST_GRAIN (x, y, z);

      /* No luck either? Give up, kill everything */
      DESPERATE_CASE (z) else DESPERATE_CASE (y) else DESPERATE_CASE (x);
#undef DESPERATE_CASE
#undef TRY_LEAST_GRAIN
#undef TRY_HALVE
    }

  /* We now have the largest possible local work-group size that satisfies
   * all the hard constraints (divide global, per-dimension bound, overall
   * bound) and our soft constraint of being as close as possible a
   * multiple of the preferred work-group size multiple. Such a greedy
   * algorithm minimizes the total number of work-groups. In moderate-sized
   * launch grid, this may result in less work-groups than the number of
   * compute units, with a resulting imbalance in the workload
   * distribution. At the same time, we want to avoid too many work-groups,
   * since some devices are penalized by such fragmentation. Finding a good
   * balance between the two is a hard problem, and generally depends on
   * the device as well as the kernel utilization of its resources.
   * Lacking that, as a first step we will simply try to guarantee that we
   * have at least one work-group per CU, as long as the local work size
   * does not drop below a given threshold.
   */

  /* Pick a minimum work-group size of 4 times the preferred work-group
   * size multiple, under the assumption that this would be a good
   * candidate below which a Compute Unit will not do enough work.
   */
  const size_t min_group_size
      = min (4 * preferred_wg_multiple, max_group_size);

  /* We need the number of Compute Units in the device, since we want
   * at least that many work-groups, if possible */

  cl_uint ncus = dev->max_compute_units;

  /* number of workgroups */
  size_t nwg_x = global_x / *local_x;
  size_t nwg_y = global_y / *local_y;
  size_t nwg_z = global_z / *local_z;

  size_t splits; /* number of splits to bring ngws to reach ncu */
  /* Only proceed if splitting wouldn't bring us below the minimum
   * group size */
  while (((splits = ncus / (nwg_x * nwg_y * nwg_z)) > 1)
         && (*local_x * *local_y * *local_z >= splits * min_group_size))
    {
      /* Very simple splitting approach: find a dimension divisible by
       * split, and lacking that divide by something less, if possible.
       * If we fail at splitting at all, we will try killing the smaller of
       * the dimensions.
       * We will set splits to 0 if we succeed in the TRY_SPLIT, so that
       * we know that we can skip the rest.
       * If we get to the end of the while without splitting and without
       * killing a dimension, we bail out early because it means we
       * couldn't do anything useful without dropping below min_group_size.
       */

#define TRY_SPLIT(coord)                                                      \
  if ((*local_##coord % splits) == 0                                          \
      && divide_chain (grain_##coord, *local_##coord / splits,                \
                       global_##coord))                                       \
    {                                                                         \
      *local_##coord /= splits;                                               \
      nwg_##coord *= splits;                                                  \
      splits = 0;                                                             \
      continue;                                                               \
    }

#define TRY_LEAST_DIM(c1, c2, c3)                                             \
  if (*local_##c1 > 1 && *local_##c1 <= *local_##c2                           \
      && *local_##c1 <= *local_##c3                                           \
      && *local_##c2 * *local_##c3 >= min_group_size)                         \
    {                                                                         \
      *local_##c1 = 1;                                                        \
      nwg_##c1 = global_##c1;                                                 \
      continue;                                                               \
    }

      while (splits > 1)
        {
          TRY_SPLIT (z) else TRY_SPLIT (y) else TRY_SPLIT (x) else splits--;
        }
      /* When we get here, splits will be 0 if we split, 1 if we failed:
       * in which case we will just kill one of the dimensions instead,
       * using the same TRY_LEAST_GRAIN and DESPERATE_CASE seen before
       */
      if (splits == 0)
        continue;

      TRY_LEAST_DIM (z, x, y)
      else TRY_LEAST_DIM (y, z, x) else TRY_LEAST_DIM (x, y, z) else break;
#undef TRY_LEAST_DIM
#undef TRY_SPLIT
    }
}

void
pocl_wg_utilization_maximizer (cl_device_id dev, cl_kernel kernel,
                               unsigned device_i,
                               size_t global_x, size_t global_y,
                               size_t global_z, size_t *local_x,
                               size_t *local_y, size_t *local_z)
{
  size_t max_group_size = dev->max_work_group_size;
  *local_x = *local_y = *local_z = 1;
  /* First check if we can trivially create a simple 1D local dimensions
     by using the max group size in one of the dimensions. */
  if (global_x % max_group_size == 0
      && max_group_size <= dev->max_work_item_sizes[0])
    *local_x = max_group_size;
  else if (global_y % max_group_size == 0
           && max_group_size <= dev->max_work_item_sizes[1])
    *local_y = max_group_size;
  else if (global_z % max_group_size == 0
           && max_group_size <= dev->max_work_item_sizes[2])
    *local_z = max_group_size;

  /* Then, perform an exhaustive search to find the dimensions with maximal
     lane utilization. */
  for (size_t z_c = 1; *local_x * *local_y * *local_z < max_group_size
                       && z_c <= min (dev->max_work_item_sizes[2], global_z);
       ++z_c)
    if (global_z % z_c == 0)
      for (size_t y_c = 1; y_c <= min (dev->max_work_item_sizes[1], global_y);
           ++y_c)
        if (global_y % y_c == 0)
          for (size_t x_c = min (dev->max_work_item_sizes[0], global_x);
               x_c >= 1; --x_c)
            if (global_x % x_c == 0)
              if (x_c * y_c * z_c <= max_group_size
                  && x_c * y_c * z_c > *local_x * *local_y * *local_z)
                {
                  *local_x = x_c;
                  *local_y = y_c;
                  *local_z = z_c;
                }
}
