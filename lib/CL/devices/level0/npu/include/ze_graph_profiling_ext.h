/*
*
* Copyright (C) 2021-2023 Intel Corporation
*
* SPDX-License-Identifier: MIT
*
*/
#ifndef _ZE_GRAPH_PROFILING_EXT_H
#define _ZE_GRAPH_PROFILING_EXT_H
#if defined(__cplusplus)
#pragma once
#endif

#ifndef ZE_PROFILING_DATA_EXT_NAME
#define ZE_PROFILING_DATA_EXT_NAME "ZE_extension_profiling_data"
#endif

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported Profiling Data Extension versions
/// 
/// @details
///     - Profiling data extension versions contain major and minor attributes, use
///       ::ZE_MAJOR_VERSION and ::ZE_MINOR_VERSION
typedef enum _ze_profiling_data_ext_version_t
{
    ZE_PROFILING_DATA_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),                ///< version 1.0
    ZE_PROFILING_DATA_EXT_VERSION_CURRENT = ZE_PROFILING_DATA_EXT_VERSION_1_0,  ///< latest known version
    ZE_PROFILING_DATA_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_profiling_data_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines structure types
typedef enum _ze_structure_type_profiling_data_ext_t
{
    ZE_STRUCTURE_TYPE_DEVICE_PROFILING_DATA_PROPERTIES = 0x1,   ///< ::ze_device_profiling_data_properties_t

} ze_structure_type_profiling_data_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Profiling Data Processing Type
typedef enum _ze_graph_profiling_type_t
{
    ZE_GRAPH_PROFILING_LAYER_LEVEL = 0x1,
    ZE_GRAPH_PROFILING_TASK_LEVEL = 0x2,
    ZE_GRAPH_PROFILING_RAW = 0x4,

    ZE_GRAPH_PROFILING_FORCE_UINT32 = 0x7fffffff

} ze_graph_profiling_type_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_GRAPH_PROFILING_LAYER_NAME
#define ZE_MAX_GRAPH_PROFILING_LAYER_NAME 256
#endif

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_GRAPH_PROFILING_LAYER_TYPE
#define ZE_MAX_GRAPH_PROFILING_LAYER_TYPE 50
#endif

///////////////////////////////////////////////////////////////////////////////
typedef enum _ze_layer_status_t
{
    ZE_LAYER_STATUS_NOT_RUN = 1,
    ZE_LAYER_STATUS_OPTIMIZED_OUT,
    ZE_LAYER_STATUS_EXECUTED

} ze_layer_status_t;

///////////////////////////////////////////////////////////////////////////////
typedef struct _ze_profiling_layer_info
{
    char name[ZE_MAX_GRAPH_PROFILING_LAYER_NAME];
    char layer_type[ZE_MAX_GRAPH_PROFILING_LAYER_TYPE];

    ze_layer_status_t status;
    uint64_t start_time_ns;   ///< Absolute start time
    uint64_t duration_ns;     ///< Total duration (from start time until last compute task completed)
    uint32_t layer_id;        ///< Not used
    uint64_t fused_layer_id;  ///< Not used

    // Aggregate compute time  (aka. "CPU" time, will include DPU, SW, DMA)
    uint64_t dpu_ns;
    uint64_t sw_ns;
    uint64_t dma_ns;

} ze_profiling_layer_info;

///////////////////////////////////////////////////////////////////////////////
typedef enum _ze_task_execute_type_t
{
    ZE_TASK_EXECUTE_NONE = 0,
    ZE_TASK_EXECUTE_DPU,
    ZE_TASK_EXECUTE_SW,
    ZE_TASK_EXECUTE_DMA
    
} ze_task_execute_type_t;

///////////////////////////////////////////////////////////////////////////////
typedef struct _ze_profiling_task_info
{
    char name[ZE_MAX_GRAPH_PROFILING_LAYER_NAME];
    char layer_type[ZE_MAX_GRAPH_PROFILING_LAYER_TYPE];

    ze_task_execute_type_t exec_type;
    uint64_t start_time_ns;
    uint64_t duration_ns;
    uint32_t active_cycles;
    uint32_t stall_cycles;
    uint32_t task_id;
    uint32_t parent_layer_id;  ///< Not used

} ze_profiling_task_info;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device profiling data properties
typedef struct _ze_device_profiling_data_properties_t
{
    ze_structure_type_profiling_data_ext_t stype;       ///< [in] type of this structure
    void* pNext;                                        ///< [in,out][optional] must be null or a pointer to an extension-specific
    ze_profiling_data_ext_version_t extensionVersion;   ///< [out] profiling data extension version

} ze_device_profiling_data_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's graph object
typedef struct _ze_graph_handle_t *ze_graph_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's graph profiling pool object
typedef struct _ze_graph_profiling_pool_handle_t *ze_graph_profiling_pool_handle_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's graph profiling query object
typedef struct _ze_graph_profiling_query_handle_t *ze_graph_profiling_query_handle_t;

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetProfilingDataProperties_ext_t)(
    ze_device_handle_t hDevice,                                             ///< [in] handle of the device
    ze_device_profiling_data_properties_t* pDeviceProfilingDataProperties   ///< [out] query result for profiling data properties of the device
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphProfilingPoolCreate_ext_t)(
    ze_graph_handle_t hGraph,                                   ///< [in] handle of the graph object
    uint32_t count,                                             ///< [in] requested count of slots in pool
    ze_graph_profiling_pool_handle_t* phProfilingPool           ///< [out] pointer to handle of profiling pool created
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphProfilingPoolDestroy_ext_t)(
    ze_graph_profiling_pool_handle_t hProfilingPool            ///< [in] handle of profiling pool to destroy
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphProfilingQueryCreate_ext_t)(
    ze_graph_profiling_pool_handle_t hProfilingPool,            ///< [in] handle of the profiling pool
    uint32_t index,                                             ///< [in] slot index to create the query from the pool
    ze_graph_profiling_query_handle_t* phProfilingQuery         ///< [out] pointer to handle of profiling pool created
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphProfilingQueryDestroy_ext_t)(
    ze_graph_profiling_query_handle_t hProfilingQuery           ///< [in] handle of profiling query to destroy
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphProfilingQueryGetData_ext_t)(
    ze_graph_profiling_query_handle_t hProfilingQuery,          ///< [in] handle of profiling query to process
    ze_graph_profiling_type_t profilingType,                    ///< [in] type of profiling requested
    uint32_t* pSize,                                            ///< [in,out] pointer to the size of the processed data
    uint8_t* pData                                              ///< [in] pointer to buffer to return processed data
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphProfilingLogGetString_ext_t)(
    ze_graph_profiling_query_handle_t phProfilingQuery,         ///< [in] handle of the graph object
    uint32_t* pSize,                                            ///< [in,out] pointer to the size of the error message
    char* pProfilingLog                                         ///< [in] pointer to buffer to return error message
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of graph profiling functions pointers
typedef struct _ze_graph_profiling_dditable_ext_t
{
    ze_pfnGraphProfilingPoolCreate_ext_t            pfnProfilingPoolCreate;
    ze_pfnGraphProfilingPoolDestroy_ext_t           pfnProfilingPoolDestroy;
    ze_pfnGraphProfilingQueryCreate_ext_t           pfnProfilingQueryCreate;
    ze_pfnGraphProfilingQueryDestroy_ext_t          pfnProfilingQueryDestroy;
    ze_pfnGraphProfilingQueryGetData_ext_t          pfnProfilingQueryGetData;
    ze_pfnDeviceGetProfilingDataProperties_ext_t    pfnDeviceGetProfilingDataProperties;
    ze_pfnGraphProfilingLogGetString_ext_t          pfnProfilingLogGetString;

} ze_graph_profiling_dditable_ext_t;

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZE_GRAPH_PROFILING_EXT_H
