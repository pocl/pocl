// Evolve the scalar wave equation with Dirichlet boundaries

/* This kernel is very short. To run efficiently, probably the
   following optimizations need to occur:
   - Vectorization (with the device's natural vector length)
   - Maybe: Loop unrolling with small 3D blocks
   - Small explicit 3D loops (to amortize stencil loads, aka "loop
     blocking")
   - Multi-threading (aka parallelization)
   - Hoist setup operations (mostly integer operations) out of the
     kernel loop
   None of these are implemented explicitly here. We could provide
   several optimized versions, and then compare with pocl's
   capabilities.
 */

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif


typedef struct grid_t {
  double dt;                    // time step
  double dx, dy, dz;            // resolution
  int ai, aj, ak;               // allocated size
  int ni, nj, nk;               // used size
} grid_t;

kernel void
scalarwave(global double       *restrict const phi,
           global double const *restrict const phi_p,
           global double const *restrict const phi_p_p,
           constant grid_t     *restrict const grid)
{
  /* printf("dt=%g\n", grid->dt); */
  /* printf("dxyz=[%g,%g,%g]\n", grid->dx, grid->dy, grid->dz); */
  /* printf("aijk=[%d,%d,%d]\n", grid->ai, grid->aj, grid->ak); */
  /* printf("nijk=[%d,%d,%d]\n", grid->ni, grid->nj, grid->nk); */
  
  double const dt = grid->dt;
  
  double const dx = grid->dx;
  double const dy = grid->dy;
  double const dz = grid->dz;
  
  double const dt2 = pown(dt,2);
  
  double const idx2 = pown(dx,-2);
  double const idy2 = pown(dy,-2);
  double const idz2 = pown(dz,-2);
  
  size_t const ai = grid->ai;
  size_t const aj = grid->aj;
  size_t const ak = grid->ak;
  
  size_t const ni = grid->ni;
  size_t const nj = grid->nj;
  size_t const nk = grid->nk;
  
  size_t const di = 1;
  size_t const dj = di * ai;
  size_t const dk = dj * aj;
  
#if 0
  printf("work_dim     =%u\n", get_work_dim());
  printf("global_size  =[%zu,%zu,%zu]\n", get_global_size(0), get_global_size(1), get_global_size(2));
  printf("global_id    =[%zu,%zu,%zu]\n", get_global_id(0), get_global_id(1), get_global_id(2));
  printf("local_size   =[%zu,%zu,%zu]\n", get_local_size(0), get_local_size(1), get_local_size(2));
  printf("local_id     =[%zu,%zu,%zu]\n", get_local_id(0), get_local_id(1), get_local_id(2));
  printf("num_groups   =[%zu,%zu,%zu]\n", get_num_groups(0), get_num_groups(1), get_num_groups(2));
  printf("group_id     =[%zu,%zu,%zu]\n", get_group_id(0), get_group_id(1), get_group_id(2));
  printf("global_offset=[%zu,%zu,%zu]\n", get_global_offset(0), get_global_offset(1), get_global_offset(2));
#endif
  
  size_t const i = get_global_id(0);
  size_t const j = get_global_id(1);
  size_t const k = get_global_id(2);
  
  // If outside the domain, do nothing
  if (__builtin_expect(i>=ni || j>=nj || k>=nk, false)) return;
  
  size_t const ind3d = di*i + dj*j + dk*k;
  
  if (__builtin_expect(i==0 || j==0 || k==0 || i==ni-1 || j==nj-1 || k==nk-1,
                       false))
  {
    // Boundary condition
    
    phi[ind3d] = 0.0;
    
  } else {
    // Scalar wave equation
    
    phi[ind3d] =
      2.0 * phi_p[ind3d] - phi_p_p[ind3d] +
      dt2 * ((phi_p[ind3d-di] - 2.0*phi_p[ind3d] + phi_p[ind3d+di]) * idx2 +
             (phi_p[ind3d-dj] - 2.0*phi_p[ind3d] + phi_p[ind3d+dj]) * idy2 +
             (phi_p[ind3d-dk] - 2.0*phi_p[ind3d] + phi_p[ind3d+dk]) * idz2);
    
  }
}
