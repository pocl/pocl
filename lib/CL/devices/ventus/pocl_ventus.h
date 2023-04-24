/**
 * @file ventus.h
 *
 * Ventus GPGPU include file
 */

#ifndef POCL_VENTUS_H
#define POCL_VENTUS_H

#include "pocl_cl.h"
#include "ventus.h"
#include "prototypes.inc"

#ifdef __cplusplus
extern "C" {
#endif

GEN_PROTOTYPES (ventus)

typedef struct vt_device_data_t {
//#if !defined(ENABLE_LLVM)
  vt_device_h vt_device;
//#endif


  #define MAX_KERNELS 16

  // allocate 1MB OpenCL print buffer
  #define PRINT_BUFFER_SIZE (1024 * 1024)


  /* List of commands ready to be executed */
  _cl_command_node *ready_list;
  /* List of commands not yet ready to be executed */
  _cl_command_node *command_list;
  /* Lock for command list related operations */
  pocl_lock_t cq_lock;

  /* Currently loaded kernel. */
  cl_kernel current_kernel;
  
  /* printf buffer */
  void *printf_buffer;
}vt_device_data_t;
typedef struct meta_data{  // 这个metadata是供驱动使用的，而不是给硬件的
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
}meta_data;

void pocl_ventus_init_device_ops(struct pocl_device_ops *ops);
char *pocl_ventus_build_hash (cl_device_id device);
unsigned int pocl_ventus_probe(struct pocl_device_ops *ops);
cl_int pocl_ventus_init (unsigned j, cl_device_id dev, const char* parameters);
void pocl_ventus_run (void *data, _cl_command_node *cmd);
cl_int pocl_ventus_uninit (unsigned j, cl_device_id device);
void ventus_command_scheduler (struct vt_device_data_t *d);
void pocl_ventus_submit (_cl_command_node *node, cl_command_queue cq);
void pocl_ventus_flush (cl_device_id device, cl_command_queue cq);
void pocl_ventus_join (cl_device_id device, cl_command_queue cq);
void pocl_ventus_notify (cl_device_id device, cl_event event, cl_event finished);

void
pocl_ventus_compile_kernel (_cl_command_node *cmd, cl_kernel kernel,
                           cl_device_id device, int specialize);
void pocl_ventus_free(cl_device_id device, cl_mem memobj);
cl_int
pocl_ventus_alloc_mem_obj(cl_device_id device, cl_mem mem_obj, void *host_ptr);
void pocl_ventus_read(void *data,
                      void *__restrict__ host_ptr,
                      pocl_mem_identifier *src_mem_id,
                      cl_mem src_buf,
                      size_t offset, 
                      size_t size);
void pocl_ventus_write(void *data,
                       const void *__restrict__ host_ptr,
                       pocl_mem_identifier *dst_mem_id,
                       cl_mem dst_buf,
                       size_t offset, 
                       size_t size);

#ifdef __cplusplus
}
#endif

#endif /* POCL_VENTUS_H */
