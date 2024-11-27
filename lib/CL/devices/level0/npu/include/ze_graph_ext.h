// Copied from https://github.com/intel/level-zero-npu-extensions.
// clang-format off
/*
*
* Copyright (C) 2021-2023 Intel Corporation
*
* SPDX-License-Identifier: MIT
*
*/
#ifndef _ZE_GRAPH_EXT_H
#define _ZE_GRAPH_EXT_H
#if defined(__cplusplus)
#pragma once
#endif

#include "ze_api.h"
#include "ze_graph_profiling_ext.h"

#ifndef ZE_GRAPH_EXT_NAME
#define ZE_GRAPH_EXT_NAME "ZE_extension_graph"
#endif

#if defined(__cplusplus)
extern "C" {
#endif

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported Graph Extension versions
///
/// @details
///     - Graph extension versions contain major and minor attributes, use
///       ::ZE_MAJOR_VERSION and ::ZE_MINOR_VERSION
typedef enum _ze_graph_ext_version_t
{
    ZE_GRAPH_EXT_VERSION_1_0 = ZE_MAKE_VERSION( 1, 0 ),         ///< version 1.0
    ZE_GRAPH_EXT_VERSION_1_1 = ZE_MAKE_VERSION( 1, 1 ),         ///< version 1.1
    ZE_GRAPH_EXT_VERSION_1_2 = ZE_MAKE_VERSION( 1, 2 ),         ///< version 1.2
    ZE_GRAPH_EXT_VERSION_1_3 = ZE_MAKE_VERSION( 1, 3 ),         ///< version 1.3
    ZE_GRAPH_EXT_VERSION_1_4 = ZE_MAKE_VERSION( 1, 4 ),         ///< version 1.4
    ZE_GRAPH_EXT_VERSION_1_5 = ZE_MAKE_VERSION( 1, 5 ),         ///< version 1.5
    ZE_GRAPH_EXT_VERSION_1_6 = ZE_MAKE_VERSION( 1, 6 ),         ///< version 1.6
    ZE_GRAPH_EXT_VERSION_1_7 = ZE_MAKE_VERSION( 1, 7 ),         ///< version 1.7
    ZE_GRAPH_EXT_VERSION_1_8 = ZE_MAKE_VERSION( 1, 8 ),         ///< version 1.8
    ZE_GRAPH_EXT_VERSION_1_9 = ZE_MAKE_VERSION( 1, 9 ),         ///< version 1.9
    ZE_GRAPH_EXT_VERSION_CURRENT = ZE_GRAPH_EXT_VERSION_1_9,    ///< latest known version
    ZE_GRAPH_EXT_VERSION_FORCE_UINT32 = 0x7fffffff

} ze_graph_ext_version_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Supported graph creation input formats
typedef enum _ze_graph_format_t
{
    ZE_GRAPH_FORMAT_NATIVE = 0x1,                   ///< Format is pre-compiled blob (elf, flatbuffers)
    ZE_GRAPH_FORMAT_NGRAPH_LITE = 0x2,              ///< Format is ngraph lite IR

} ze_graph_format_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Compiler version information
typedef struct _ze_graph_compiler_version_info_t
{
    uint16_t major;
    uint16_t minor;

} ze_graph_compiler_version_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Defines structure types
typedef enum _ze_structure_type_graph_ext_t
{
    ZE_STRUCTURE_TYPE_DEVICE_GRAPH_PROPERTIES = 0x1,                    ///< ::ze_device_graph_properties_t
    ZE_STRUCTURE_TYPE_GRAPH_DESC_PROPERTIES = 0x2,                      ///< ::ze_graph_desc_t
    ZE_STRUCTURE_TYPE_GRAPH_PROPERTIES = 0x3,                           ///< ::ze_graph_properties_t
    ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_PROPERTIES = 0x4,                  ///< ::ze_graph_argument_properties_t
    ZE_STRUCTURE_TYPE_GRAPH_ACTIVATION_KERNEL = 0x5,                    ///< ::ze_graph_activation_kernel_t
    ZE_STRUCTURE_TYPE_GRAPH_ARGUMENT_METADATA = 0x6,                    ///< ::ze_graph_argument_metadata_t
    ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_ARGUMENT_EXP_DESC_DEPRECATED = 0x7, ///< ::ze_mutable_graph_argument_exp_desc_t
    ZE_STRUCTURE_TYPE_MUTABLE_GRAPH_PROFILING_QUERY_EXP_DESC = 0x8      ///< ::ze_mutable_graph_profiling_query_exp_desc_t

} ze_structure_type_graph_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device graph properties
typedef struct _ze_device_graph_properties_t
{
    ze_structure_type_graph_ext_t stype;                ///< [in] type of this structure
    void* pNext;                                        ///< [in,out][optional] must be null or a pointer to an extension-specific
    ze_graph_ext_version_t graphExtensionVersion;       ///< [out] graph extension version
    ze_graph_compiler_version_info_t compilerVersion;   ///< [out] compiler version
    ze_graph_format_t graphFormatsSupported;            ///< [out] graph formats supported
    uint32_t maxOVOpsetVersionSupported;                ///< [out] max OV opset version supported by the compiler

} ze_device_graph_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef enum _ze_graph_argument_precision_t
{
    ZE_GRAPH_ARGUMENT_PRECISION_UNKNOWN = 0x00,

    ZE_GRAPH_ARGUMENT_PRECISION_FP64 = 0x0F,
    ZE_GRAPH_ARGUMENT_PRECISION_FP32 = 0x01,
    ZE_GRAPH_ARGUMENT_PRECISION_FP16 = 0x02,
    ZE_GRAPH_ARGUMENT_PRECISION_BF16 = 0x09,

    ZE_GRAPH_ARGUMENT_PRECISION_UINT64 = 0x10,
    ZE_GRAPH_ARGUMENT_PRECISION_UINT32 = 0x0A,
    ZE_GRAPH_ARGUMENT_PRECISION_UINT16 = 0x03,
    ZE_GRAPH_ARGUMENT_PRECISION_UINT8 = 0x04,
    ZE_GRAPH_ARGUMENT_PRECISION_UINT4 = 0x0B,

    ZE_GRAPH_ARGUMENT_PRECISION_INT64 = 0x011,
    ZE_GRAPH_ARGUMENT_PRECISION_INT32 = 0x05,
    ZE_GRAPH_ARGUMENT_PRECISION_INT16 = 0x06,
    ZE_GRAPH_ARGUMENT_PRECISION_INT8 = 0x07,
    ZE_GRAPH_ARGUMENT_PRECISION_INT4 = 0x0C,

    ZE_GRAPH_ARGUMENT_PRECISION_BIN = 0x08,
    ZE_GRAPH_ARGUMENT_PRECISION_DYNAMIC = 0x0D,
    ZE_GRAPH_ARGUMENT_PRECISION_BOOLEAN = 0x0E,
    

} ze_graph_argument_precision_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef enum _ze_graph_argument_layout_t
{
    ZE_GRAPH_ARGUMENT_LAYOUT_ANY        = 0x00,

    ZE_GRAPH_ARGUMENT_LAYOUT_NCHW,
    ZE_GRAPH_ARGUMENT_LAYOUT_NHWC,
    ZE_GRAPH_ARGUMENT_LAYOUT_NCDHW,
    ZE_GRAPH_ARGUMENT_LAYOUT_NDHWC,

    ZE_GRAPH_ARGUMENT_LAYOUT_OIHW       = 0x40,

    ZE_GRAPH_ARGUMENT_LAYOUT_C          = 0x60,

    ZE_GRAPH_ARGUMENT_LAYOUT_CHW        = 0x80,

    ZE_GRAPH_ARGUMENT_LAYOUT_HW         = 0xC0,
    ZE_GRAPH_ARGUMENT_LAYOUT_NC,
    ZE_GRAPH_ARGUMENT_LAYOUT_CN,

    ZE_GRAPH_ARGUMENT_LAYOUT_BLOCKED    = 0xC8

} ze_graph_argument_layout_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Activation Shave Desc (passed through ze_graph_desc pNext)
typedef struct _ze_activation_kernel_desc_t
{
    ze_structure_type_graph_ext_t stype;            ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] must be null or a pointer to an extension-specific
    size_t kernelDataSize;                          ///< [in] Size of kernel data buffer in bytes
    const uint8_t* pKernelData;                     ///< [in] Pointer to kernel data buffer

} ze_activation_kernel_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph descriptor
typedef struct _ze_graph_desc_t
{
    ze_structure_type_graph_ext_t stype;            ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] must be null or a pointer to an extension-specific
    ze_graph_format_t format;                       ///< [in] Graph format passed in with input
    size_t inputSize;                               ///< [in] Size of input buffer in bytes
    const uint8_t* pInput;                          ///< [in] Pointer to input buffer
    const char* pBuildFlags;                        ///< [in][optional] Null terminated string containing build flags. Options:
                                                    ///< - '--inputs_precisions="<arg>:<precision> <arg2>:<precision> ..."'
                                                    ///<   '--outputs_precisions="<arg>:<precision> <arg2>:<precision> ..."'
                                                    ///<   - Set input and output arguments precision. Supported precisions:
                                                    ///<     FP64, FP32, FP16, BF16, U64, U32, U16, U8, U4, I64, I32, I16, I8, I4, BIN
                                                    ///< - '--inputs_layouts="<arg>:<layout> <arg2>:<layout> ..."'
                                                    ///<   '--outputs_layouts="<arg>:<layout> <arg2>:<layout> ..."'
                                                    ///<   - Set input and output arguments layout. Supported layouts:
                                                    ///<     NCHW, NHWC, NCDHW, NDHWC, OIHW, C, CHW, HW, NC, CN
                                                    ///< - '--config PARAM="VALUE" PARAM2="VALUE" ...'
                                                    ///<   - compile options string passed directly to compiler
} ze_graph_desc_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Example ze_graph_desc pBuildFlags
///
/// --inputs_precisions="in1:U8" --inputs_layouts="in1:NCHW" --outputs_precisions="out1:FP16 out2:FP16" --outputs_layouts="out1:NCHW out2:NCHW" --config PERFORMANCE_HINT="THROUGHPUT"
///
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph properties
typedef struct _ze_graph_properties_t
{
    ze_structure_type_graph_ext_t stype;            ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] must be null or a pointer to an extension-specific
    uint32_t numGraphArgs;                          ///< [out] number of graph arguments

} ze_graph_properties_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef enum _ze_graph_argument_type_t
{
    ZE_GRAPH_ARGUMENT_TYPE_INPUT,
    ZE_GRAPH_ARGUMENT_TYPE_OUTPUT

} ze_graph_argument_type_t;

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_GRAPH_ARGUMENT_NAME
/// @brief Maximum device name string size
#define ZE_MAX_GRAPH_ARGUMENT_NAME  256
#endif // ZE_MAX_GRAPH_ARGUMENT_NAME

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE
/// @brief Maximum device name string size
#define ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE 5
#endif // ZE_MAX_GRAPH_ARGUMENT_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef struct _ze_graph_argument_properties_t
{
    ze_structure_type_graph_ext_t stype;                    ///< [in] type of this structure
    void* pNext;                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
    char name[ZE_MAX_GRAPH_ARGUMENT_NAME];                  ///< [out] name from input IR
    ze_graph_argument_type_t type;                          ///< [out] type of graph argument
    uint32_t dims[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE];   ///< [out] tensor dimensions upto 5D
    ze_graph_argument_precision_t networkPrecision;         ///< [out] precision from input IR
    ze_graph_argument_layout_t networkLayout;               ///< [out] layout from input IR
    ze_graph_argument_precision_t devicePrecision;          ///< [out] precision from compiled executable
    ze_graph_argument_layout_t deviceLayout;                ///< [out] layout from compiled executable

} ze_graph_argument_properties_t;

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetGraphProperties_ext_t)(
    ze_device_handle_t hDevice,                           ///< [in] handle of the device
    ze_device_graph_properties_t *pDeviceGraphProperties  ///< [out] query result for graph properties of the device
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphCreate_ext_t)(
    ze_context_handle_t hContext,                   ///< [in] handle of the context
    ze_device_handle_t hDevice,                     ///< [in] handle of the device
    const ze_graph_desc_t* desc,                    ///< [in] pointer to graph descriptor
    ze_graph_handle_t* phGraph                      ///< [out] pointer to handle of graph object created
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphDestroy_ext_t)(
    ze_graph_handle_t hGraph                        ///< [in][release] handle of graph object to destroy
    );

//////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphGetNativeBinary_ext_t)(
    ze_graph_handle_t hGraph,                       ///< [in] handle of the graph object
    size_t* pSize,                                  ///< [in,out] size of native binary in bytes.
    uint8_t* pGraphNativeBinary                     ///< [in,out][optional] byte pointer to native binary
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphGetProperties_ext_t)(
    ze_graph_handle_t hGraph,                       ///< [in] handle of the graph object
    ze_graph_properties_t* pGraphProperties         ///< [in,out] query result for graph properties.
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphGetArgumentProperties_ext_t)(
    ze_graph_handle_t hGraph,                                       ///< [in] handle of the graph object
    uint32_t argIndex,                                              ///< [in] index of the argument to get properties
    ze_graph_argument_properties_t* pGraphArgumentProperties        ///< [in,out] query result for graph argument properties.
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphSetArgumentValue_ext_t)(
    ze_graph_handle_t hGraph,                       ///< [in] handle of the graph object
    uint32_t argIndex,                              ///< [in] index of the argument
    const void* pArgValue                           ///< [in] value to bind to the index
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnAppendGraphInitialize_ext_t)(
    ze_command_list_handle_t hCommandList,          ///< [in] handle of the command list
    ze_graph_handle_t hGraph,                       ///< [in] handle of the graph
    ze_event_handle_t hSignalEvent,                 ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                         ///< [in][optional] number of events to wait on before launching
    ze_event_handle_t* phWaitEvents                 ///< [in][optional] handle of the events to wait on before launching
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnAppendGraphExecute_ext_t)(
    ze_command_list_handle_t hCommandList,              ///< [in] handle of the command list
    ze_graph_handle_t hGraph,                           ///< [in] handle of the graph
    ze_graph_profiling_query_handle_t hProfilingQuery,  ///< [in][optional] handle of profiling query
    ze_event_handle_t hSignalEvent,                     ///< [in][optional] handle of the event to signal on completion
    uint32_t numWaitEvents,                             ///< [in][optional] number of events to wait on before launching
    ze_event_handle_t* phWaitEvents                     ///< [in][optional] handle of the events to wait on before launching
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension version 1.1

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_GRAPH_TENSOR_REF_DIMS
/// @brief Maximum tensor reference dimensions size
#define ZE_MAX_GRAPH_TENSOR_REF_DIMS 8
#endif // ZE_MAX_GRAPH_TENSOR_REF_DIMS

///////////////////////////////////////////////////////////////////////////////
#ifndef ZE_MAX_GRAPH_TENSOR_NAMES_SIZE
/// @brief Maximum tensor names size
#define ZE_MAX_GRAPH_TENSOR_NAMES_SIZE 32
#endif // ZE_MAX_GRAPH_TENSOR_NAMES_SIZE

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument properties
typedef enum _ze_graph_metadata_type
{
    ZE_GRAPH_METADATA_TYPE_UNDEFINED = 0,
    ZE_GRAPH_METADATA_TYPE_DYNAMIC = 1,
    ZE_GRAPH_METADATA_TYPE_BOOLEAN = 2,
    ZE_GRAPH_METADATA_TYPE_BF16 = 3,
    ZE_GRAPH_METADATA_TYPE_F16 = 4,
    ZE_GRAPH_METADATA_TYPE_F32 = 5,
    ZE_GRAPH_METADATA_TYPE_F64 = 6,
    ZE_GRAPH_METADATA_TYPE_I4 = 7,
    ZE_GRAPH_METADATA_TYPE_I8 = 8,
    ZE_GRAPH_METADATA_TYPE_I16 = 9,
    ZE_GRAPH_METADATA_TYPE_I32 = 10,
    ZE_GRAPH_METADATA_TYPE_I64 = 11,
    ZE_GRAPH_METADATA_TYPE_U1 = 12,
    ZE_GRAPH_METADATA_TYPE_U4 = 13,
    ZE_GRAPH_METADATA_TYPE_U8 = 14,
    ZE_GRAPH_METADATA_TYPE_U16 = 15,
    ZE_GRAPH_METADATA_TYPE_U32 = 16,
    ZE_GRAPH_METADATA_TYPE_U64 = 17

} ze_graph_metadata_type;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph argument metadata
typedef struct _ze_graph_argument_metadata_t
{
    ze_structure_type_graph_ext_t stype;                                           ///< [in] type of this structure
    void* pNext;                                                                   ///< [in,out][optional] must be null or a pointer to an extension-specific
    ze_graph_argument_type_t type;                                                 ///< [out] type of argument
    char friendly_name[ZE_MAX_GRAPH_ARGUMENT_NAME];                                ///< [out] friendly name
    ze_graph_metadata_type data_type;                                              ///< [out] data type of argument
    uint64_t shape[ZE_MAX_GRAPH_TENSOR_REF_DIMS];                                  ///< [out] tensor shape
    uint32_t shape_size;                                                           ///< [out] size of shape array
    char tensor_names[ZE_MAX_GRAPH_TENSOR_NAMES_SIZE][ZE_MAX_GRAPH_ARGUMENT_NAME]; ///< [out] tensor name array
    uint32_t tensor_names_count;                                                   ///< [out] size of tensor name array
    char input_name[ZE_MAX_GRAPH_ARGUMENT_NAME];                                   ///< [out] input name

} ze_graph_argument_metadata_t;

///////////////////////////////////////////////////////////////////////////////
typedef struct _ze_graph_argument_properties_2_t
{
    ze_structure_type_graph_ext_t stype;                    ///< [in] type of this structure
    void* pNext;                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
    char name[ZE_MAX_GRAPH_ARGUMENT_NAME];                  ///< [out] name from input IR
    ze_graph_argument_type_t type;                          ///< [out] type of graph argument
    uint32_t dims[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE];   ///< [out] tensor dimensions upto 5D
    ze_graph_argument_precision_t networkPrecision;         ///< [out] precision from input IR
    ze_graph_argument_layout_t networkLayout;               ///< [out] layout from input IR
    ze_graph_argument_precision_t devicePrecision;          ///< [out] precision from compiled executable
    ze_graph_argument_layout_t deviceLayout;                ///< [out] layout from compiled executable

    // version 2
    float quantReverseScale;                                ///< [out] Quantized tensor reverse scale value for input argument
    uint8_t quantZeroPoint;                                 ///< [out] Quantized tesnor zero point value for input argument

} ze_graph_argument_properties_2_t;

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphGetArgumentProperties_ext_2_t)(
    ze_graph_handle_t hGraph,                                       ///< [in] handle of the graph object
    uint32_t argIndex,                                              ///< [in] index of the argument to get properties
    ze_graph_argument_properties_2_t* pGraphArgumentProperties      ///< [in,out] query result for graph argument properties.
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphGetArgumentMetadata_ext_t)(
    ze_graph_handle_t hGraph,                            ///< [in] handle of the graph object
    uint32_t argIndex,                                   ///< [in] index of the argument to get metadata
    ze_graph_argument_metadata_t* pGraphArgumentMetadata ///< [in,out] query result for graph argument metadata
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension version 1.2

///////////////////////////////////////////////////////////////////////////////
typedef struct _ze_graph_argument_properties_3_t
{
    ze_structure_type_graph_ext_t stype;                    ///< [in] type of this structure
    void* pNext;                                            ///< [in,out][optional] must be null or a pointer to an extension-specific
    char name[ZE_MAX_GRAPH_ARGUMENT_NAME];                  ///< [out] name from input IR
    ze_graph_argument_type_t type;                          ///< [out] type of graph argument
    uint32_t dims[ZE_MAX_GRAPH_ARGUMENT_DIMENSIONS_SIZE];   ///< [out] tensor dimensions upto 5D
    ze_graph_argument_precision_t networkPrecision;         ///< [out] precision from input IR
    ze_graph_argument_layout_t networkLayout;               ///< [out] layout from input IR
    ze_graph_argument_precision_t devicePrecision;          ///< [out] precision from compiled executable
    ze_graph_argument_layout_t deviceLayout;                ///< [out] layout from compiled executable

    // version 2
    float quantReverseScale;                                ///< [out] Quantized tensor reverse scale value for input argument
    uint8_t quantZeroPoint;                                 ///< [out] Quantized tesnor zero point value for input argument

    // version 3
    uint32_t dims_count;                                    ///< [out] size of shape array
    char debug_friendly_name[ZE_MAX_GRAPH_ARGUMENT_NAME];   ///< [out] debug friendly name
    char associated_tensor_names[ZE_MAX_GRAPH_TENSOR_NAMES_SIZE][ZE_MAX_GRAPH_ARGUMENT_NAME]; ///< [out] tensor name array
    uint32_t associated_tensor_names_count;                 ///< [out] size of tensor name array

} ze_graph_argument_properties_3_t;

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphGetArgumentProperties_ext_3_t)(
    ze_graph_handle_t hGraph,                                       ///< [in] handle of the graph object
    uint32_t argIndex,                                              ///< [in] index of the argument to get properties
    ze_graph_argument_properties_3_t* pGraphArgumentProperties      ///< [in,out] query result for graph argument properties.
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension version 1.3

///////////////////////////////////////////////////////////////////////////////
/// @brief Handle of driver's graph query network object
typedef struct _ze_graph_query_network_handle_t *ze_graph_query_network_handle_t;

typedef ze_result_t (ZE_APICALL *ze_pfnGraphQueryNetworkCreate_ext_t)(
    ze_context_handle_t hContext,                         ///< [in] handle of the context object
    ze_device_handle_t hDevice,                           ///< [in] handle of the device
    const ze_graph_desc_t* desc,                          ///< [in] pointer to graph descriptor
    ze_graph_query_network_handle_t* phGraphQueryNetwork  ///< [out] pointer to handle of graph query network object created
);

typedef ze_result_t (ZE_APICALL *ze_pfnGraphQueryNetworkDestroy_ext_t)(
    ze_graph_query_network_handle_t hGraphQueryNetwork  ///< [in][release] handle of the graph query network
);

typedef ze_result_t (ZE_APICALL *ze_pfnGraphQueryNetworkGetSupportedLayers_ext_t)(
    ze_graph_query_network_handle_t hGraphQueryNetwork, ///< [in] handle of the graph query network
    size_t *pSize,                                      ///< [in,out] size of supported layers string
    char *pSupportedLayers                              ///< [in,out][optional] pointer to null-terminated string of the supported layers
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension version 1.4

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphBuildLogGetString_ext_t)(
    ze_graph_handle_t hGraph,                       ///< [in] handle of the graph object
    uint32_t* pSize,                                ///< [in,out] pointer to the size of the error message
    char* pBuildLog                                 ///< [in] pointer to buffer to return error message
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension version 1.5

///////////////////////////////////////////////////////////////////////////////
/// @brief Bitfield of graph flags
typedef enum _ze_graph_flags_t
{
    ZE_GRAPH_FLAG_NONE = 0x0,
    ZE_GRAPH_FLAG_DISABLE_CACHING = 0x1,           ///< Disable driver managed caching
    ZE_GRAPH_FLAG_ENABLE_PROFILING = 0x2,          ///< Enable layer and task level timings

} ze_graph_flags_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph descriptor
typedef struct _ze_graph_desc_2_t
{
    ze_structure_type_graph_ext_t stype;            ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] must be null or a pointer to an extension-specific
    ze_graph_format_t format;                       ///< [in] Graph format passed in with input
    size_t inputSize;                               ///< [in] Size of input buffer in bytes
    const uint8_t* pInput;                          ///< [in] Pointer to input buffer
    const char* pBuildFlags;                        ///< [in][optional] Null terminated string containing build flags. Options:
                                                    ///< - '--inputs_precisions="<arg>:<precision> <arg2>:<precision> ..."'
                                                    ///<   '--outputs_precisions="<arg>:<precision> <arg2>:<precision> ..."'
                                                    ///<   - Set input and output arguments precision. Supported precisions:
                                                    ///<     FP64, FP32, FP16, BF16, U64, U32, U16, U8, U4, I64, I32, I16, I8, I4, BIN
                                                    ///< - '--inputs_layouts="<arg>:<layout> <arg2>:<layout> ..."'
                                                    ///<   '--outputs_layouts="<arg>:<layout> <arg2>:<layout> ..."'
                                                    ///<   - Set input and output arguments layout. Supported layouts:
                                                    ///<     NCHW, NHWC, NCDHW, NDHWC, OIHW, C, CHW, HW, NC, CN
                                                    ///< - '--config PARAM="VALUE" PARAM2="VALUE" ...'
                                                    ///<   - compile options string passed directly to compiler
    uint32_t flags;                                 ///< [in][optional] Graph creation flags
} ze_graph_desc_2_t;

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphCreate_ext_2_t)(
    ze_context_handle_t hContext,                   ///< [in] handle of the context
    ze_device_handle_t hDevice,                     ///< [in] handle of the device
    const ze_graph_desc_2_t* desc,                  ///< [in] pointer to graph descriptor
    ze_graph_handle_t* phGraph                      ///< [out] pointer to handle of graph object created
    );

typedef ze_result_t (ZE_APICALL *ze_pfnGraphQueryNetworkCreate_ext_2_t)(
    ze_context_handle_t hContext,                         ///< [in] handle of the context object
    ze_device_handle_t hDevice,                           ///< [in] handle of the device
    const ze_graph_desc_2_t* desc,                        ///< [in] pointer to graph descriptor
    ze_graph_query_network_handle_t* phGraphQueryNetwork  ///< [out] pointer to handle of graph query network object created
);

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph memory query types
typedef enum _ze_graph_memory_query_type_t
{
    ZE_GRAPH_QUERY_MEMORY_DDR = 0x01,               ///< DDR memory allocations

} ze_graph_memory_query_type_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph memory query
typedef struct _ze_graph_memory_query_t
{
    uint64_t total;                                 ///< total memory allowed per process for specified type (reported in bytes)
    uint64_t allocated;                             ///< context's current total memory allocated for specified type (reported in bytes)

} ze_graph_memory_query_t;

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphQueryContextMemory_ext_t)(
    ze_context_handle_t hContext,                   ///< [in] handle of the context object
    ze_graph_memory_query_type_t type,              ///< [in] type of memory query
    ze_graph_memory_query_t* query                  ///< [out] result of query
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension version 1.6

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph version information
typedef struct _ze_graph_version_info_t
{
    uint32_t major;
    uint32_t minor;
    uint32_t patch;

} ze_graph_version_info_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Device graph properties
typedef struct _ze_device_graph_properties_2_t
{
    ze_structure_type_graph_ext_t stype;                ///< [in] type of this structure
    void* pNext;                                        ///< [in,out][optional] must be null or a pointer to an extension-specific
    ze_graph_ext_version_t graphExtensionVersion;       ///< [out] graph extension version
    ze_graph_compiler_version_info_t compilerVersion;   ///< [out] compiler version
    ze_graph_format_t graphFormatsSupported;            ///< [out] graph formats supported
    uint32_t maxOVOpsetVersionSupported;                ///< [out] max OV opset version supported by the compiler
    ze_graph_version_info_t elfVersion;                 ///< [out] elf container version
    ze_graph_version_info_t runtimeVersion;             ///< [out] firmware runtime version

} ze_device_graph_properties_2_t;

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnDeviceGetGraphProperties_ext_2_t)(
    ze_device_handle_t hDevice,                             ///< [in] handle of the device
    ze_device_graph_properties_2_t *pDeviceGraphProperties  ///< [out] query result for graph properties of the device
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension version 1.7

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphGetNativeBinary_ext_2_t)(
    ze_graph_handle_t hGraph,                       ///< [in] handle of the graph object
    size_t* pSize,                                  ///< [out] size of native binary in bytes
    const uint8_t** pGraphNativeBinary              ///< [out] double pointer to view of native binary, driver owns the memory
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension version 1.8

///////////////////////////////////////////////////////////////////////////////
/// @brief Stage required to initialize the graph
typedef enum _ze_graph_init_stage_t
{
    ZE_GRAPH_STAGE_COMMAND_LIST_INITIALIZE = 0x1,   ///< Call to pfnAppendGraphInitialize is required
    ZE_GRAPH_STAGE_INITIALIZE = 0x2,                ///< Call to pfnGraphInitialize is required
    ZE_GRAPH_STAGE_FORCE_UINT32 = 0x7fffffff

} ze_graph_init_stage_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Graph properties
typedef struct _ze_graph_properties_2_t
{
    ze_structure_type_graph_ext_t stype;            ///< [in] type of this structure
    void* pNext;                                    ///< [in,out][optional] must be null or a pointer to an extension-specific
    uint32_t numGraphArgs;                          ///< [out] number of graph arguments
    ze_graph_init_stage_t initStageRequired;        ///< [out] stage required to initialize the graph

} ze_graph_properties_2_t;

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphGetProperties_ext_2_t)(
    ze_graph_handle_t hGraph,                       ///< [in] handle of the graph object
    ze_graph_properties_2_t* pGraphProperties       ///< [in,out] query result for graph properties
    );

///////////////////////////////////////////////////////////////////////////////
typedef ze_result_t (ZE_APICALL *ze_pfnGraphInitialize_ext_t)(
    ze_graph_handle_t hGraph                        ///< [in] handle of the graph
    );

///////////////////////////////////////////////////////////////////////////////
/// @brief Extension version 1.9

// Version 1.9 ze_graph_memory_query_t reports values in bytes on Linux and Windows (previously reported in KB on Windows)

///////////////////////////////////////////////////////////////////////////////
/// @brief Table of Graph functions pointers
typedef struct _ze_graph_dditable_ext_t
{
    // version 1.0
    ze_pfnGraphCreate_ext_t                     pfnCreate;
    ze_pfnGraphDestroy_ext_t                    pfnDestroy;
    ze_pfnGraphGetProperties_ext_t              pfnGetProperties;
    ze_pfnGraphGetArgumentProperties_ext_t      pfnGetArgumentProperties;
    ze_pfnGraphSetArgumentValue_ext_t           pfnSetArgumentValue;
    ze_pfnAppendGraphInitialize_ext_t           pfnAppendGraphInitialize;
    ze_pfnAppendGraphExecute_ext_t              pfnAppendGraphExecute;
    ze_pfnGraphGetNativeBinary_ext_t            pfnGetNativeBinary;
    ze_pfnDeviceGetGraphProperties_ext_t        pfnDeviceGetGraphProperties;

    // version 1.1
    ze_pfnGraphGetArgumentMetadata_ext_t        pfnGraphGetArgumentMetadata;
    ze_pfnGraphGetArgumentProperties_ext_2_t    pfnGetArgumentProperties2;

    // version 1.2
    ze_pfnGraphGetArgumentProperties_ext_3_t    pfnGetArgumentProperties3;

    // version 1.3
    ze_pfnGraphQueryNetworkCreate_ext_t             pfnQueryNetworkCreate;
    ze_pfnGraphQueryNetworkDestroy_ext_t            pfnQueryNetworkDestroy;
    ze_pfnGraphQueryNetworkGetSupportedLayers_ext_t pfnQueryNetworkGetSupportedLayers;

    // version 1.4
    ze_pfnGraphBuildLogGetString_ext_t          pfnBuildLogGetString;

    // version 1.5
    ze_pfnGraphCreate_ext_2_t                   pfnCreate2;
    ze_pfnGraphQueryNetworkCreate_ext_2_t       pfnQueryNetworkCreate2;
    ze_pfnGraphQueryContextMemory_ext_t         pfnQueryContextMemory;

    // version 1.6
    ze_pfnDeviceGetGraphProperties_ext_2_t      pfnDeviceGetGraphProperties2;

    // version 1.7
    ze_pfnGraphGetNativeBinary_ext_2_t          pfnGetNativeBinary2;

    // version 1.8
    ze_pfnGraphGetProperties_ext_2_t            pfnGetProperties2;
    ze_pfnGraphInitialize_ext_t                 pfnGraphInitialize;

} ze_graph_dditable_ext_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Mutable command lists NPU specific flags and structures
typedef enum _ze_mutable_command_npu_exp_flag_t
{
    ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_ARGUMENT = ZE_BIT(6),         ///< graph argument
    ZE_MUTABLE_COMMAND_EXP_FLAG_GRAPH_PROFILING_QUERY = ZE_BIT(7),  ///< graph profiling query

} ze_mutable_command_npu_exp_flag_t;

///////////////////////////////////////////////////////////////////////////////
/// @brief Mutable graph profiling query descriptor
typedef struct _ze_mutable_graph_profiling_query_exp_desc_t
{
    ze_structure_type_graph_ext_t stype;                ///< [in] type of this structure
    const void* pNext;                                  ///< [in][optional] must be null or a pointer to an extension-specific
                                                        ///< structure (i.e. contains stype and pNext).
    uint64_t commandId;                                 ///< [in] command identifier
    ze_graph_profiling_query_handle_t hProfilingQuery;  ///< [in] handle of profiling query

} ze_mutable_graph_profiling_query_exp_desc_t;

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // _ZE_GRAPH_EXT_H
