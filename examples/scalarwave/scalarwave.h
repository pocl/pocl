#ifndef SCALARWAVE_H
#define SCALARWAVE_H

#define GRID_GRANULARITY 1

typedef struct grid_t {
  cl_double dt;                 // time step
  cl_double dx, dy, dz;         // resolution
  cl_int ai, aj, ak;            // allocated size
  cl_int ni, nj, nk;            // used size
} grid_t;
 
int 
exec_scalarwave_kernel (const char *program_source, 
                        cl_double       *restrict phi,
                        cl_double const *restrict phi_p,
                        cl_double const *restrict phi_p_p,
                        grid_t    const *restrict grid);

#endif  // #ifndef SCALARWAVE_H
