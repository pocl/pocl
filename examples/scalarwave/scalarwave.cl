// Evolve the scalar wave equation with Dirichlet boundaries

typedef struct grid_t {
  double dt;                    // time step
  double dx, dy, dz;            // resolution
  size_t ai, aj, ak;            // allocated size
  size_t ni, nj, nk;            // used size
} grid_t;

kernel void
scalarwave(global double       *restrict const phi,
           global double const *restrict const phi_p,
           global double const *restrict const phi_p_p,
           constant grid_t     *restrict const grid)
{
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
  
  size_t const i = get_global_id(0);
  size_t const j = get_global_id(1);
  size_t const k = get_global_id(2);
  
  // If outside the domain, do nothing
  if (i>=ni || j>=nj || k>=nk) return;
  
  size_t const ind3d = di*i + dj*j + dk*k;
  if (i==0 || i==ni-1 || j==0 || j==nj-1 || k==0 || k==nk-1) {
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
