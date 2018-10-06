/* Original code:
 * The MIT License (MIT)
 * Copyright (c) 2014 SURFsara
 * https://github.com/CNugteren/myGEMM/blob/master/src/kernels.cl
 */

#define ITYPE uint

// Simple version
__kernel void
myGEMM2 (const __global float *A, const __global float *B, __global float *C,
         uint M, uint N, uint K)

{
  // Thread identifiers
  const ITYPE globalRow = get_global_id (0); // Row ID of C (0..M)
  const ITYPE globalCol = get_global_id (1); // Col ID of C (0..N)

  // Compute a single element (loop over K)
  float acc = 0.0f;
  for (ITYPE k = 0; k < K; k++)
    {
#ifdef USE_FMA
      acc = fma (A[k * M + globalRow], B[globalCol * K + k], acc);
#else
      acc += A[k * M + globalRow] * B[globalCol * K + k];
#endif
    }

  // Store the result
  C[globalCol * M + globalRow] = acc;
}

/**********************************************************************/
/**********************************************************************/
/**********************************************************************/

#ifdef MYGEMM4

#define TS (LOCAL_SIZE)
/* work per thread */
#define WPT (LOCAL_SIZE / 4)

// TS/WPT == RTS
#define RTS 4

// Tiled and coalesced version
__kernel void
myGEMM4 (const __global float *A, const __global float *B, __global float *C,
         uint M, uint N, uint K)
{

  // Thread identifiers
  const ITYPE row = get_local_id (0); // Local row ID (max: TS)
  const ITYPE col = get_local_id (1); // Local col ID (max: TS/WPT == RTS)
  const ITYPE globalRow = TS * get_group_id (0) + row; // Row ID of C (0..M)
  const ITYPE globalCol = TS * get_group_id (1) + col; // Col ID of C (0..N)

  // Local memory to fit a tile of TS*TS elements of A and B
  __local float Asub[TS][TS];
  __local float Bsub[TS][TS];

  // Initialise the accumulation registers
  float acc[WPT];
  for (ITYPE w = 0; w < WPT; w++)
    {
      acc[w] = 0.0f;
    }

  // Loop over all tiles
  const ITYPE numTiles = K / TS;
  for (ITYPE t = 0; t < numTiles; t++)
    {

      // Load one tile of A and B into local memory
      for (ITYPE w = 0; w < WPT; w++)
        {
          const ITYPE tiledRow = TS * t + row;
          const ITYPE tiledCol = TS * t + col;
          Asub[col + w * RTS][row] = A[(tiledCol + w * RTS) * M + globalRow];
          Bsub[col + w * RTS][row] = B[(globalCol + w * RTS) * K + tiledRow];
        }

      // Synchronise to make sure the tile is loaded
      barrier (CLK_LOCAL_MEM_FENCE);

      // Perform the computation for a single tile
      for (ITYPE k = 0; k < TS; k++)
        {
          for (ITYPE w = 0; w < WPT; w++)
            {
#ifdef USE_FMA
              acc[w] = fma (Asub[k][row], Bsub[col + w * RTS][k], acc[w]);
#else
              acc[w] += Asub[k][row] * Bsub[col + w * RTS][k];
#endif
            }
        }

      // Synchronise before loading the next tile
      barrier (CLK_LOCAL_MEM_FENCE);
    }

  // Store the final results in C
  for (ITYPE w = 0; w < WPT; w++)
    {
      C[(globalCol + w * RTS) * M + globalRow] = acc[w];
    }
}

#endif

/**********************************************************************/
/**********************************************************************/
/**********************************************************************/

#define TRANSPOSEX 8
#define TRANSPOSEY 8

// Simple transpose kernel for a P * Q matrix
__kernel void
transpose (const ITYPE P, const ITYPE Q, const __global float *input,
           __global float *output)
{

  // Thread identifiers
  const ITYPE tx = get_local_id (0);
  const ITYPE ty = get_local_id (1);
  const ITYPE ID0 = get_group_id (0) * TRANSPOSEX + tx; // 0..P
  const ITYPE ID1 = get_group_id (1) * TRANSPOSEY + ty; // 0..Q

  // Set-up the local memory for shuffling
  __local float buffer[TRANSPOSEX][TRANSPOSEY];

  // Swap the x and y coordinates to perform the rotation (coalesced)
  //    if (ID0 < P && ID1 < Q) {
  buffer[ty][tx] = input[ID1 * P + ID0];
  //    }

  // Synchronise all threads
  barrier (CLK_LOCAL_MEM_FENCE);

  // We don't have to swap the x and y thread indices here,
  // because that's already done in the local memory
  const ITYPE newID0 = get_group_id (1) * TRANSPOSEY + tx;
  const ITYPE newID1 = get_group_id (0) * TRANSPOSEX + ty;

  // Store the transposed result (coalesced)
  //    if (newID0 < Q && newID1 < P) {
  output[newID1 * Q + newID0] = buffer[tx][ty];
  //    }
}

/**********************************************************************/
/**********************************************************************/
/**********************************************************************/

#ifdef MYGEMM6

#ifndef TSM
#error TSM must be defined
#endif
#ifndef TSN
#error TSN must be defined
#endif
#ifndef TSK
#error TSK must be defined
#endif

#define WPTM 8 // The work-per-thread in dimension M
#define WPTN 8 // The work-per-thread in dimension N

#define RTSM (TSM / WPTM) // The reduced tile-size in dimension M
#define RTSN (TSN / WPTN) // The reduced tile-size in dimension N

#define LPTA ((TSK * TSM) / (RTSM * RTSN)) // Loads-per-thread for A
#define LPTB ((TSK * TSN) / (RTSM * RTSN)) // Loads-per-thread for B

/* Original values:

#define TSM 128                // The tile-size in dimension M
#define TSN 128                // The tile-size in dimension N
#define TSK 16                 // The tile-size in dimension K
#define WPTM 8                 // The work-per-thread in dimension M
#define WPTN 8                 // The work-per-thread in dimension N
#define RTSM (TSM/WPTM)        // The reduced tile-size in dimension M
#define RTSN (TSN/WPTN)        // The reduced tile-size in dimension N
#define LPTA ((TSK*TSM)/(RTSM*RTSN)) // Loads-per-thread for A
#define LPTB ((TSK*TSN)/(RTSM*RTSN)) // Loads-per-thread for B

*/

// Use 2D register blocking (further increase in work per thread)
kernel void
myGEMM6 (const __global float *A, const __global float *B, __global float *C,
         uint M, uint N, uint K)
{
  // Thread identifiers
  const ITYPE tidm = get_local_id (0);          // Local row ID (max: TSM/WPTM)
  const ITYPE tidn = get_local_id (1);          // Local col ID (max: TSN/WPTN)
  const ITYPE offsetM = TSM * get_group_id (0); // Work-group offset
  const ITYPE offsetN = TSN * get_group_id (1); // Work-group offset

  // Local memory to fit a tile of A and B
  __local float Asub[TSK][TSM];
  __local float Bsub[TSN][TSK];

  // Allocate register space
  float Areg;
  float Breg[WPTN];
  float acc[WPTM][WPTN];

  // Initialise the accumulation registers
  for (ITYPE wm = 0; wm < WPTM; wm++)
    {
      for (ITYPE wn = 0; wn < WPTN; wn++)
        {
          acc[wm][wn] = 0.0f;
        }
    }

  // Loop over all tiles
  ITYPE numTiles = K / TSK;
  for (ITYPE t = 0; t < numTiles; t++)
    {

      // Load one tile of A and B into local memory
      for (ITYPE la = 0; la < LPTA; la++)
        {
          ITYPE tid = tidn * RTSM + tidm;
          ITYPE id = la * RTSN * RTSM + tid;
          ITYPE row = id % TSM;
          ITYPE col = id / TSM;
          ITYPE tiledIndex = TSK * t + col;
          Asub[col][row] = A[tiledIndex * M + offsetM + row];
          Bsub[row][col] = B[tiledIndex * N + offsetN + row];
        }

      // Synchronise to make sure the tile is loaded
      barrier (CLK_LOCAL_MEM_FENCE);

      // Loop over the values of a single tile
      for (ITYPE k = 0; k < TSK; k++)
        {

          // Cache the values of Bsub in registers
          for (ITYPE wn = 0; wn < WPTN; wn++)
            {
              ITYPE col = tidn + wn * RTSN;
              Breg[wn] = Bsub[col][k];
            }

          // Perform the computation
          for (ITYPE wm = 0; wm < WPTM; wm++)
            {
              ITYPE row = tidm + wm * RTSM;
              Areg = Asub[k][row];
              for (ITYPE wn = 0; wn < WPTN; wn++)
                {
#ifdef USE_FMA
                  acc[wm][wn] = fma (Areg, Breg[wn], acc[wm][wn]);
#else
                  acc[wm][wn] += Areg * Breg[wn];
#endif
                }
            }
        }

      // Synchronise before loading the next tile
      barrier (CLK_LOCAL_MEM_FENCE);
    }

  // Store the final results in C
  for (ITYPE wm = 0; wm < WPTM; wm++)
    {
      ITYPE globalRow = offsetM + tidm + wm * RTSM;
      for (ITYPE wn = 0; wn < WPTN; wn++)
        {
          ITYPE globalCol = offsetN + tidn + wn * RTSN;
          C[globalCol * M + globalRow] = acc[wm][wn];
        }
    }
}

#endif
