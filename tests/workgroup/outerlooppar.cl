/**
 * Test the outer loop parallelization using the implicit loop barriers
 * added by pocl.
 *
 * The barrier should be inserted inside a barrierless loop in case pocl
 * can analyze it's safe to do so. The cases are tested here.
 */
__kernel void
test_kernel (void)
{
  int gid = get_global_id (0);
  int lid = get_local_id (0);

  if (lid == 0)
    printf ("vertical:\n");
  /* This loop cannot be horizontally vectorized by the implicit loop barrier
     mechanism because of an iteration count that depends on the gid. */
  for (int i = 0; i < gid; ++i) {
    printf ("i: %d gid: %d\n", i, gid);
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
  if (lid == 0)
    printf ("horizontal:\n");
  /* This loop should be horizontally vectorized because the iteration count
     does not depend on the gid.*/
#pragma nounroll
  for (int i = 0; i < get_local_size(0); ++i) {
    if (i < 4)
      printf ("i: %d gid: %d\n", i, gid);
  }

  barrier(CLK_GLOBAL_MEM_FENCE);
  if (lid == 0)
    printf ("vertical:\n");

  /* This loop should not be horizontally vectorized because the loop
     is entered only by the subset of the work items.*/
  if (gid > 0) {
    for (int i = 0; i < get_local_size(0); ++i) {
      printf ("i: %d gid: %d\n", i, gid);
    }
  }
  barrier(CLK_GLOBAL_MEM_FENCE);
}
