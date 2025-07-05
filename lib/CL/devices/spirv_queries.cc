/* spirv_queries.cc - sets up SPIR-V queries for the device
 *
 * Copyright (c) 2025 Ben Ashbaugh
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <vector>

#include "spirv.hh"
#include "spirv_queries.h"

#ifdef __cplusplus
extern "C" {
#endif

void pocl_setup_spirv_queries(cl_device_id dev) {
  if (dev->num_ils_with_version == 0)
    return;

  std::vector<const char *> SpirvExtendedInstructionSets;
  std::vector<const char *> SpirvExtensions;
  std::vector<cl_uint> SpirvCapabilities;

  SpirvExtendedInstructionSets.push_back("OpenCL.std");

  SpirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Addresses));
  SpirvCapabilities.push_back(
      static_cast<cl_uint>(spv::Capability::Float16Buffer));
  SpirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Int16));
  SpirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Int8));
  SpirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Kernel));
  SpirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Linkage));
  SpirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Vector16));

  if (dev->has_64bit_long) {
    SpirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Int64));
  }

  if (dev->image_support) {
    SpirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Image1D));
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::ImageBasic));
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::ImageBuffer));
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::LiteralSampler));
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::Sampled1D));
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SampledBuffer));
  }

  // TODO: SPIR-V 1.6 support and Capability::UniformDecoration

  if (dev->max_read_write_image_args != 0) {
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::ImageReadWrite));
  }

  if (dev->generic_as_support) {
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GenericPointer));
  }

  if (dev->max_num_sub_groups != 0 || dev->wg_collective_func_support) {
    SpirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Groups));
  }

  if (dev->pipe_support) {
    SpirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Pipes));
  }

  if (dev->max_num_sub_groups != 0) {
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SubgroupDispatch));
  }

  if (strstr(dev->extensions, "cl_khr_expect_assume")) {
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::ExpectAssumeKHR));
  }

  if (strstr(dev->extensions, "cl_khr_extended_bit_ops")) {
    SpirvExtensions.push_back("SPV_KHR_bit_instructions");
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::BitInstructions));
  }

  if (strstr(dev->extensions, "cl_khr_fp16")) {
    SpirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Float16));
  }

  if (strstr(dev->extensions, "cl_khr_fp64")) {
    SpirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Float64));
  }

  if (strstr(dev->extensions, "cl_khr_int64_base_atomics") ||
      strstr(dev->extensions, "cl_khr_int64_extended_atomics")) {
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::Int64Atomics));
  }

  if (strstr(dev->extensions, "cl_khr_integer_dot_product")) {
    SpirvExtensions.push_back("SPV_KHR_integer_dot_product");
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::DotProduct));
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::DotProductInput4x8BitPacked));
    if (dev->dot_product_caps &
        CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR) {
      SpirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::DotProductInput4x8Bit));
    }
  }

  if (strstr(dev->extensions, "cl_khr_kernel_clock")) {
    SpirvExtensions.push_back("SPV_KHR_shader_clock");
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::ShaderClockKHR));
  }

  if (strstr(dev->extensions, "cl_khr_mipmap_image") &&
      strstr(dev->extensions, "cl_khr_mipmap_image_writes")) {
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::ImageMipmap));
  }

  if (strstr(dev->extensions, "cl_khr_spirv_linkonce_odr")) {
    SpirvExtensions.push_back("SPV_KHR_linkonce_odr");
  }

  if (strstr(dev->extensions, "cl_khr_spirv_no_integer_wrap_decoration")) {
    SpirvExtensions.push_back("SPV_KHR_no_integer_wrap_decoration");
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_ballot")) {
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformBallot));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_clustered_reduce")) {
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformClustered));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_named_barrier")) {
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::NamedBarrier));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_non_uniform_arithmetic")) {
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformArithmetic));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_non_uniform_vote")) {
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniform));
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformVote));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_rotate")) {
    SpirvExtensions.push_back("SPV_KHR_subgroup_rotate");
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformRotateKHR));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_shuffle")) {
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformShuffle));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_shuffle_relative")) {
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformShuffleRelative));
  }

  if (strstr(dev->extensions, "cl_khr_work_group_uniform_arithmetic")) {
    SpirvExtensions.push_back("SPV_KHR_uniform_group_instructions");
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupUniformArithmeticKHR));
  }

  if (strstr(dev->extensions, "cl_ext_float_atomics")) {
    if (dev->single_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)) {
      SpirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::AtomicFloat32AddEXT));
    }
    if (dev->single_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)) {
      SpirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::AtomicFloat32MinMaxEXT));
    }
    if (dev->half_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
                                    CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)) {
      SpirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::AtomicFloat16AddEXT));
    }
    if (dev->half_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                    CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)) {
      SpirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::AtomicFloat16MinMaxEXT));
    }
    if (dev->double_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)) {
      SpirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::AtomicFloat64AddEXT));
    }
    if (dev->double_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)) {
      SpirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::AtomicFloat64MinMaxEXT));
    }
    if (dev->single_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT) ||
        dev->double_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)) {
      SpirvExtensions.push_back("SPV_EXT_shader_atomic_float_add");
    }
    if (dev->single_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT) ||
        dev->half_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                    CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT) ||
        dev->double_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)) {
      SpirvExtensions.push_back("SPV_EXT_shader_atomic_float_min_max");
    }
  }

  if (strstr(dev->extensions, "cl_intel_spirv_subgroups")) {
    SpirvExtensions.push_back("SPV_INTEL_subgroups");
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SubgroupBufferBlockIOINTEL));
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SubgroupImageBlockIOINTEL));
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SubgroupShuffleINTEL));
  }

  if (strstr(dev->extensions, "cl_intel_split_work_group_barrier")) {
    SpirvExtensions.push_back("SPV_INTEL_split_barrier");
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SplitBarrierINTEL));
  }

  if (strstr(dev->extensions, "cl_intel_subgroup_buffer_prefetch")) {
    SpirvExtensions.push_back("SPV_INTEL_subgroup_buffer_prefetch");
    SpirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SubgroupBufferPrefetchINTEL));
  }

  if (!SpirvExtendedInstructionSets.empty()) {
    dev->num_spirv_extended_instruction_sets =
        SpirvExtendedInstructionSets.size();
    dev->spirv_extended_instruction_sets = (const char **)malloc(
        SpirvExtendedInstructionSets.size() * sizeof(const char *));
    memcpy(dev->spirv_extended_instruction_sets,
           SpirvExtendedInstructionSets.data(),
           SpirvExtendedInstructionSets.size() * sizeof(const char *));
  }

  if (!SpirvExtensions.empty()) {
    dev->num_spirv_extensions = SpirvExtensions.size();
    dev->spirv_extensions =
        (const char **)malloc(SpirvExtensions.size() * sizeof(const char *));
    memcpy(dev->spirv_extensions, SpirvExtensions.data(),
           SpirvExtensions.size() * sizeof(const char *));
  }

  if (!SpirvCapabilities.empty()) {
    dev->num_spirv_capabilities = SpirvCapabilities.size();
    cl_uint *temp =
        (cl_uint *)malloc(SpirvCapabilities.size() * sizeof(cl_uint));
    memcpy(temp, SpirvCapabilities.data(),
           SpirvCapabilities.size() * sizeof(cl_uint));
    dev->spirv_capabilities = temp;
  }
}

#ifdef __cplusplus
}
#endif
