
#include "stdbool.h"

#include "dnn_cpu_ref.h"
#include "pocl_onednn.h"

#define USE_ONEDNN

static void
generateStrides (const int *dimA, int *strideA, int nbDims)
{
  strideA[nbDims - 1] = 1;
  for (int d = nbDims - 2; d >= 0; d--)
    {
      strideA[d] = strideA[d + 1] * dimA[d + 1];
    }
}

// Convert a linear index
// i = d_1 s_1 ... s_n + d_2 s_2 ... s_n + d_n-1 s_n + d_n
// into a multidimensional index
// (d_1, d_2, ..., d_n)
void
lin2dim (int id, int *ids, const int *dims, int length)
{
  int idrem = id;
  int prod = 1; // accumulates the product of the dimensions
  for (int i = length - 1; i >= 0; i--)
    {
      ids[i] = (idrem / prod) % dims[i];
      idrem = id - ids[i] * prod;
      prod *= dims[i];
    }
}

// Convert a multidimensional index
// (d_1, d_2, ..., d_n)
// into a linear index
// i = d_1 s_1 + ... + d_n s_n
static int
dim2lin (const int *ids, const int *strides, int length)
{
  int res = 0;
  for (int i = 0; i < length; i++)
    {
      res += ids[i] * strides[i];
    }
  return res;
}

static float
doFma (float fval, float ival, float tmp)
{
  return fval * ival + tmp;
}

static void
doEpilog (float *out, int idx, float alphaAcc, float beta)
{
  if (beta == 0.f)
    {
      out[idx] = alphaAcc;
    }
  else
    {
      out[idx] = alphaAcc + out[idx] * beta;
    }
}

static inline int
getFwdConvDilatedFilterDim (int filterDim, int dilation)
{
  return ((filterDim - 1) * dilation) + 1;
}

static inline int
getFwdConvPaddedImageDim (int tensorDim, int pad)
{
  return tensorDim + (2 * pad);
}

static inline int
getFwdConvOutputDim (int tensorDim, int pad, int filterDim, int stride,
                     int dilation)
{
  int p = (getFwdConvPaddedImageDim (tensorDim, pad)
           - getFwdConvDilatedFilterDim (filterDim, dilation))
              / stride
          + 1;
  return (p);
}

static void
conv_cpu_ref (const float *inputData, const float *filterData,
              float *outputData, float alpha, float beta, const int *inDims,
              const int *filDims, const int *outDims, const int *inStride,
              const int *outStride, const int *stride, const int *pad,
              const int *dilation)
{
  int imDims = 4 - 2;

  int filStride[8] = { 0 };
  generateStrides (filDims, filStride, 4);

  // Number of pixels in output
  int nPixelsOut = 1;
  for (int i = 2; i < 4; i++)
    {
      nPixelsOut *= outDims[i];
    }
  // Number of pixels in filter
  int nPixelsFil = 1;
  for (int i = 2; i < 4; i++)
    {
      nPixelsFil *= filDims[i];
    }
  // Used to store coordinates
  int filIds[8] = { 0 };
  int outIds[8] = { 0 };
  int inIds[8] = { 0 };
  int tmpIds[8] = { 0 };
  // For each image in the output
  for (int ni = 0; ni < outDims[0]; ni++)
    {
      // For each outer feature layer of the output image
      for (int ki_outer = 0; ki_outer < outDims[1]; ki_outer++)
        {
          int outputOffset = ni * outStride[0] + ki_outer * outStride[1];
          // For every pixel in this output image's feature layer
          for (int outId = 0; outId < nPixelsOut; outId++)
            {
              // Get output pixel ids
              lin2dim (outId, outIds, outDims + 2,
                       imDims); // Skip n and k dimensions
              // Now we get the coordinates in input space of the "top left"
              // corner of the filter: multiply by stride and remove pad
              for (int d = 0; d < imDims; d++)
                {
                  inIds[d] = outIds[d] * stride[d] - pad[d];
                }
              // For each inner feature layer of the output image
              for (int ki_inner = 0; ki_inner < 1; ki_inner++)
                {
                  // We prepare to accumulate
                  float tmp = 0;
                  // For each outer feature layer of the input image and filter
                  for (int ci = 0; ci < inDims[1]; ci++)
                    {
                      int inputOffset = ni * inStride[0] + ci * inStride[1];
                      int filterOffset = (ki_outer + ki_inner) * filStride[0]
                                         + ci * filStride[1];
                      // Now for every pixel in the filter
                      for (int filId = 0; filId < nPixelsFil; filId++)
                        {
                          // Get the position of the pixel
                          lin2dim (filId, filIds, filDims + 2, imDims);
                          // Compute the corresponding output pixel
                          // and check whether we are in the padding area on
                          // the fly too (not that for convolution, we flip the
                          // image patch; equivalent to flipping the filter
                          // patch).
                          bool inside = true;
                          for (int d = 0; d < imDims && inside; d++)
                            {
                              tmpIds[d]
                                  = inIds[d]
                                    + dilation[d]
                                          * (filDims[2 + d] - 1 - filIds[d]);
                              // If we are in the padding area: stop and skip
                              // computations
                              inside &= (tmpIds[d] >= 0
                                         && tmpIds[d] < inDims[2 + d]);
                            }
                          if (inside)
                            {
                              int actualTmpId
                                  = inputOffset
                                    + dim2lin (tmpIds, (inStride) + 2, imDims);
                              // int actualFilId = filterOffset + filId ;
                              int actualFilId
                                  = filterOffset
                                    + dim2lin (filIds, (filStride) + 2,
                                               imDims);

                              // For each inner feature layer of the input
                              // image and filter
                              for (int i = 0; i < 1; i++)
                                {
                                  float fval = filterData[actualFilId + i];
                                  float ival = inputData[actualTmpId + i];
                                  tmp = doFma (fval, ival, tmp);
                                }
                            }
                        }
                    }
                  // Store final result in proper position in output image
                  int actualOutId
                      = outputOffset
                        + dim2lin (outIds, (outStride) + 2, imDims);
                  doEpilog (outputData, actualOutId + ki_inner, alpha * tmp,
                            beta);
                }
            }
        }
    }
}

void
submit_dnn_builtin_kernel (_cl_command_node *cmd)
{
  _cl_command_run run = cmd->command.run;
  cl_kernel kernel = run.kernel;
  cl_program prog = kernel->program;
  pocl_argument *arguments = run.arguments;
  struct pocl_context pc = run.pc;
  pocl_kernel_metadata_t *meta = kernel->meta;
  cl_device_id device = cmd->device;

  // Input, weight and output buffers come from pocl's buffer management
  cl_mem mem = *(void **)arguments[0].value;
  float *in_data = (float *)(mem->device_ptrs[device->global_mem_id].mem_ptr
                             + arguments[0].offset);

  mem = *(void **)arguments[1].value;
  float *filt_data = (float *)(mem->device_ptrs[device->global_mem_id].mem_ptr
                               + arguments[1].offset);

  mem = *(void **)arguments[2].value;
  float *out_data = (float *)(mem->device_ptrs[device->global_mem_id].mem_ptr
                              + arguments[2].offset);

  // All the other convolution dimensions are passed as arguments.
  int in_n = *(int *)(arguments[3].value);
  int in_c = *(int *)(arguments[4].value);
  int in_h = *(int *)(arguments[5].value);
  int in_w = *(int *)(arguments[6].value);
  int inDims[4] = { in_n, in_c, in_h, in_w };

  int filt_k = *(int *)(arguments[7].value);
  int filt_c = *(int *)(arguments[8].value);
  int filt_h = *(int *)(arguments[9].value);
  int filt_w = *(int *)(arguments[10].value);
  int filtDims[4] = { filt_k, filt_c, filt_h, filt_w };

  int str_h = *(int *)(arguments[11].value);
  int str_w = *(int *)(arguments[12].value);
  int dil_h = *(int *)(arguments[13].value);
  int dil_w = *(int *)(arguments[14].value);
  int pad_h = *(int *)(arguments[15].value);
  int pad_w = *(int *)(arguments[16].value);
  int pad[2] = { pad_h, pad_w };
  int dilation[2] = { dil_h, dil_w };
  int stride[2] = { str_h, str_w };

  float alpha = *(float *)(arguments[17].value);
  float beta = *(float *)(arguments[18].value);

  POCL_MSG_PRINT_INFO ("ARGS:%p,%p,%p in:%i,%i,%i,%i filt %i,%i,%i,%i "
                       "strdilpad %i,%i,%i,%i,%i,%i\n",
                       in_data, filt_data, out_data, in_n, in_c, in_h, in_w,
                       filt_k, filt_c, filt_h, filt_w, str_h, str_w, dil_h,
                       dil_w, pad_h, pad_w);

  int inStride[4];
  int outStride[4];
  int outDims[4];

  outDims[0] = inDims[0];
  outDims[1] = filtDims[0];
  for (int dim = 0; dim < 2; dim++)
    {
      outDims[dim + 2]
          = getFwdConvOutputDim (inDims[dim + 2], pad[dim], filtDims[dim + 2],
                                 stride[dim], dilation[dim]);
    }

  generateStrides (inDims, inStride, 4);
  generateStrides (outDims, outStride, 4);

#ifdef USE_ONEDNN
  conv_onecnn (in_data, filt_data, out_data, alpha, beta, inDims, filtDims,
               outDims, inStride, outStride, stride, pad, dilation);
#else
  conv_cpu_ref (in_data, filt_data, out_data, alpha, beta, inDims, filtDims,
                outDims, inStride, outStride, stride, pad, dilation);
#endif
}