
#ifndef POCL_ONEDNN_H
#define POCL_ONEDNN_H

void conv_onecnn (const float *inputData, const float *filterData,
                  float *outputData, float alpha, float beta,
                  const int *inDims, const int *filDims, const int *outDims,
                  const int *inStride, const int *outStride, const int *stride,
                  const int *pad, const int *dilation);

#endif
