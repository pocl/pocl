/* scalarwave - Scalar wave evolution.

   Copyright (c) 2011 Universidad Rey Juan Carlos
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <CL/opencl.h>
#include "scalarwave.h"

#define NT 4                    // time steps
#define NX 33                   // grid size
// NX, rounded up
#define AX ((NX + GRID_GRANULARITY-1) / GRID_GRANULARITY * GRID_GRANULARITY)

int
main (void)
{
  FILE *source_file = fopen("scalarwave.cl", "r");
  if (source_file == NULL) 
    source_file = fopen (SRCDIR "/scalarwave.cl", "r");

  assert(source_file != NULL && "scalarwave.cl not found!");

  fseek (source_file, 0, SEEK_END);
  int const source_size = ftell (source_file);
  fseek (source_file, 0, SEEK_SET);

  char *source = malloc (source_size + 1);
  assert (source != NULL);

  fread (source, source_size, 1, source_file);
  source[source_size] = '\0';

  fclose (source_file);
  
  grid_t grid;
  grid.dt = 0.5/(NX-1);
  grid.dx = grid.dy = grid.dz = 1.0/(NX-1);
  grid.ai = grid.aj = grid.ak = AX;
  grid.ni = grid.nj = grid.nk = NX;

  cl_double *restrict phi     =
    malloc (grid.ai*grid.aj*grid.ak * sizeof *phi    );
  cl_double *restrict phi_p   =
    malloc (grid.ai*grid.aj*grid.ak * sizeof *phi_p  );
  cl_double *restrict phi_p_p =
    malloc (grid.ai*grid.aj*grid.ak * sizeof *phi_p_p);

  // Set up initial data (TODO: do this on the device as well)
  double const kx = 2*M_PI;
  double const ky = 2*M_PI;
  double const kz = 2*M_PI;
  double const omega = sqrt(pow(kx,2)+pow(ky,2)+pow(kz,2));
  for (int k = 0; k < NX; ++k) {
    for (int j = 0; j < NX; ++j) {
      for (int i = 0; i < NX; ++i) {
        double const t0 =   0.0;
        double const t1 =  -grid.dt;
        double const x  = i*grid.dx;
        double const y  = j*grid.dy;
        double const z  = k*grid.dz;
        int const ind3d = i+grid.ai*(j+grid.aj*k);
        phi  [ind3d] = cos(kx*x) * cos(ky*y) * cos(kz*z) * cos(omega*t0);
        phi_p[ind3d] = cos(kx*x) * cos(ky*y) * cos(kz*z) * cos(omega*t1);
      }
    }
  }

  // Take some time steps
  for (int n=0; n<NT; ++n) {
    printf ("Time step %d: t=%g\n", n, n*grid.dt);
    
    // Cycle time levels
    {
      cl_double *tmp = phi_p_p;
      phi_p_p = phi_p;
      phi_p = phi;
      phi = tmp;
    }

    int const ierr =
      exec_scalarwave_kernel (source, phi, phi_p, phi_p_p, &grid);
    assert(!ierr);

  }
  
  for (int i=0; i<NX; ++i) {
    int const j = i;
    int const k = i;
    double const x = grid.dx*i;
    double const y = grid.dy*j;
    double const z = grid.dz*k;
    int const ind3d = i*grid.ai*(j+grid.aj*k);
    
    printf ("phi[%g,%g,%g] = %g\n", x,y,z, phi[ind3d]);
  }

  printf ("OK\n");

  return 0;
}
