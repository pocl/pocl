/**
 * @file ventus.h
 *
 * Ventus GPGPU include file
 */

#ifndef POCL_VENTUS_H
#define POCL_VENTUS_H

#include "pocl_cl.h"

#include "prototypes.inc"
GEN_PROTOTYPES (ventus)

struct meta_data{  // 这个metadata是供驱动使用的，而不是给硬件的
    uint64_t kernel_id;
    uint64_t kernel_size[3];///> 每个kernel的workgroup三维数目
    uint64_t wf_size; ///> 每个warp的thread数目
    uint64_t wg_size; ///> 每个workgroup的warp数目
    uint64_t metaDataBaseAddr;///> CSR_KNL的值，
    uint64_t ldsSize;///> 每个workgroup使用的local memory的大小
    uint64_t pdsSize;///> 每个thread用到的private memory大小
    uint64_t sgprUsage;///> 每个workgroup使用的标量寄存器数目
    uint64_t vgprUsage;///> 每个thread使用的向量寄存器数目
    uint64_t pdsBaseAddr;///> private memory的基址，要转成每个workgroup的基地址， wf_size*wg_size*pdsSize
};



#endif /* POCL_VENTUS_H */
