/**
 * @file ventus.h
 * @brief this file defined some ventus driver API when running OpenCL program
 * @author yangzexia
 * @version 1.0
 * @date 2022-11-16
 * 
 * @copyright Tsinghua DSPLAB
 * 
 * @par 修改日志:
 * <table>
 * <tr><th>Date       <th>Version <th>Author  <th>Description
 * <tr><td>2022-11-16 <td>1.0     <td>yangzexia     <td>main程序和驱动之间的接口
 * <tr><td>2022-12-15 <td>1.1     <td>yangzexia     <td>增加了执行所有kernel并返回id的接口
 * </table>
 */
#ifndef __VT_DRIVER_H__
#define __VT_DRIVER_H__

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
//#include <queue>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* vt_device_h; ///< 类型定义，指向vt_device类的指针

typedef void* vt_buffer_h; ///< 类型定义，指向vt_buffer类的指针


#define MAX_TIMEOUT               (60*60*1000)   // 1hr 


/// @brief 【已实现】打开并连接一个GPGPU设备
/// @param hdevice 指向设备的指针
/// @return 若无错误则返回0，否则返回-1
int vt_dev_open(vt_device_h* hdevice);

/// @brief 【已实现】当所有操作完成时，关闭设备
/// @param hdevice 指向设备的指针
/// @return 若无错误则返回0，否则返回-1
int vt_dev_close(vt_device_h hdevice);

/// return device configurations
// int vt_dev_caps(vt_device_h* hdevice, uint32_t caps_id, uint64_t *value);

/// @brief 【已实现】以任务为单位，在GPGPU设备上分配虚拟内存空间（创建根页表）
/// @param hdevice 指向设备的指针
/// @param taskID context ID，需要从0开始迭加分配
/// @return 若无错误则返回0，否则返回-1
int vt_root_mem_alloc(vt_device_h hdevice, int taskID);


/// @brief 【已实现】释放任务对应的虚拟内存空间和已分配的物理空间（删除根页表）
/// @param hdevice 指向设备的指针
/// @param taskID context ID
/// @return 若无错误则返回0，否则返回-1
int vt_root_mem_free(vt_device_h hdevice, int taskID);
/// @brief 【已实现】为设备分配buffer
/// @param hdevice 指向设备的指针
/// @param size buffer大小
/// @param vaddr 保存设备端内存地址的指针，函数在分配好内存空间后，修改该指针的值为分配的内存的地址
/// @param BUF_TYPE 要分配的buffer类型 只读or读写（device端）
/// @param taskID context ID
/// @param kernelID kernel ID
/// @return 若无错误则返回0，否则返回-1
int vt_buf_alloc(vt_device_h hdevice, uint64_t size, uint64_t *vaddr, int BUF_TYPE, uint64_t taskID, uint64_t kernelID);

/// @brief 【已实现】释放buffer
/// @param hbuffer 指向设备的指针
/// @param size buffer大小
/// @param vaddr 要释放的设备端内存的起始地址
/// @param taskID context ID
/// @param kernelID kernel IDD
/// @return 若无错误则返回0，否则返回-1
int vt_buf_free(vt_device_h hdevice, uint64_t size, uint64_t *vaddr, uint64_t taskID, uint64_t kernelID);

/// @brief 【已实现】释放指定的buffer
/// @param hbuffer 指向设备的指针
/// @param size buffer大小
/// @param vaddr 要释放的设备端内存的起始地址
/// @param taskID context ID
/// @param kernelID kernel IDD
/// @return 若无错误则返回0，否则返回-1
int vt_one_buf_free(vt_device_h hdevice, uint64_t size, uint64_t *vaddr, uint64_t taskID, uint64_t kernelID);

/// @brief 【已实现】将数据从buffer复制到设备内存
/// @param hdevice 指向设备的指针
/// @param dev_vaddr 设备端保存数据的起始虚拟地址
/// @param src_addr 源数据的起始地址
/// @param size 数据大小
/// @param taskID 任务ID
/// @param kernelID kernel ID
/// @return 若无错误则返回0，否则返回-1
int vt_copy_to_dev(vt_device_h hdevice, uint64_t dev_vaddr, const void *src_addr, uint64_t size, uint64_t taskID, uint64_t kernelID);


/// @brief 【已实现】将数据从设备内存复制到buffer
/// @param hdevice 指向设备的指针
/// @param dev_vaddr 设备端保存数据的起始虚拟地址
/// @param dst_addr 数据的目标地址
/// @param size 数据大小
/// @param taskID 任务ID
/// @param kernelID kernel ID
/// @return 若无错误则返回0，否则返回-1
int vt_copy_from_dev(vt_device_h hdevice, uint64_t dev_vaddr, void *dst_addr, uint64_t size, uint64_t taskID, uint64_t kernelID);


/// @brief 【已实现】设备开始执行任务
/// @param hdevice 指向设备的指针
/// @param metaData 要执行的kernel的metaData
/// @param taskID 该kernel属于哪个context
/// @return 若无错误则返回0，否则返回-1
int vt_start(vt_device_h hdevice, void* metaData, uint64_t taskID);

/// @brief 【已实现】等待设备执行完成
/// @param hdevice 指向设备的指针
/// @param timeout 等待时间，单位为毫秒
/// @return 若无错误则返回0，否则返回-1
int vt_ready_wait(vt_device_h hdevice, uint64_t timeout);

/// @brief 【已实现】执行所有的kernel，更新已完成的kernel队列，
/// 队列的每个元素的低ceil(log2(MAX_KERNEL))是kernelID，高位是contextID
/// @param hdevice 指向设备的指针
/// @param finished_list 指向已完成kernel的list
/// @return 若无错误则返回0，否则返回-1
//int vt_finish_all_kernel(vt_device_h hdevice, std::queue<int> *finished_kernel_list);

////////////////////////////// UTILITY FUNCIONS ///////////////////////////////

/// @brief 【已实现】上传kernel文件的一部分到设备，由vt_upload_kernel_file调用
/// @param device 指向设备的指针
/// @param content 数据内容
/// @param size 数据大小
/// @param taskID 对应的任务ID
/// @return 若无错误则返回0，否则返回-1
int vt_upload_kernel_bytes(vt_device_h device, const void* content, uint64_t size, int taskID);


/// @brief 【已实现】上传kernel文件到设备
/// @param device 指向设备的指针
/// @param filename kernel文件的名称
/// @param kernelID 对应的kernel ID
/// @return 若无错误则返回0，否则返回-1
int vt_upload_kernel_file(vt_device_h device, const char* filename, int kernelID);

/// dump performance counters
/// @brief 【未实现】性能计数
/// @param device 指向设备的指针
/// @param stream 
/// @return 
int vt_dump_perf(vt_device_h device, FILE* stream);


#ifdef __cplusplus
}
#endif

#endif // __VT_DRIVER_H__
