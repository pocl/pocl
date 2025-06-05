/* spirv_queries.h - sets up SPIR-V queries for the device
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

  std::vector<const char *> spirvExtendedInstructionSets;
  std::vector<const char *> spirvExtensions;
  std::vector<cl_uint> spirvCapabilities;

  spirvExtendedInstructionSets.push_back("OpenCL.std");

  spirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Addresses));
  spirvCapabilities.push_back(
      static_cast<cl_uint>(spv::Capability::Float16Buffer));
  spirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Int16));
  spirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Int8));
  spirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Kernel));
  spirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Linkage));
  spirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Vector16));

  if (dev->has_64bit_long) {
    spirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Int64));
  }

  if (dev->image_support) {
    spirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Image1D));
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::ImageBasic));
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::ImageBuffer));
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::LiteralSampler));
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::Sampled1D));
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SampledBuffer));
  }

  // TODO: SPIR-V 1.6 support and Capability::UniformDecoration

  if (dev->max_read_write_image_args != 0) {
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::ImageReadWrite));
  }

  if (dev->generic_as_support) {
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GenericPointer));
  }

  if (dev->max_num_sub_groups != 0 || dev->wg_collective_func_support) {
    spirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Groups));
  }

  if (dev->pipe_support) {
    spirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Pipes));
  }

  if (dev->max_num_sub_groups != 0) {
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SubgroupDispatch));
  }

  if (strstr(dev->extensions, "cl_khr_expect_assume")) {
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::ExpectAssumeKHR));
  }

  if (strstr(dev->extensions, "cl_khr_extended_bit_ops")) {
    spirvExtensions.push_back("SPV_KHR_bit_instructions");
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::BitInstructions));
  }

  if (strstr(dev->extensions, "cl_khr_fp16")) {
    spirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Float16));
  }

  if (strstr(dev->extensions, "cl_khr_fp64")) {
    spirvCapabilities.push_back(static_cast<cl_uint>(spv::Capability::Float64));
  }

  if (strstr(dev->extensions, "cl_khr_int64_base_atomics") ||
      strstr(dev->extensions, "cl_khr_int64_extended_atomics")) {
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::Int64Atomics));
  }

  if (strstr(dev->extensions, "cl_khr_integer_dot_product")) {
    spirvExtensions.push_back("SPV_KHR_integer_dot_product");
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::DotProduct));
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::DotProductInput4x8BitPacked));
    if (dev->dot_product_caps &
        CL_DEVICE_INTEGER_DOT_PRODUCT_INPUT_4x8BIT_KHR) {
      spirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::DotProductInput4x8Bit));
    }
  }

  if (strstr(dev->extensions, "cl_khr_kernel_clock")) {
    spirvExtensions.push_back("SPV_KHR_shader_clock");
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::ShaderClockKHR));
  }

  if (strstr(dev->extensions, "cl_khr_mipmap_image") &&
      strstr(dev->extensions, "cl_khr_mipmap_image_writes")) {
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::ImageMipmap));
  }

  if (strstr(dev->extensions, "cl_khr_spirv_linkonce_odr")) {
    spirvExtensions.push_back("SPV_KHR_linkonce_odr");
  }

  if (strstr(dev->extensions, "cl_khr_spirv_no_integer_wrap_decoration")) {
    spirvExtensions.push_back("SPV_KHR_no_integer_wrap_decoration");
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_ballot")) {
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformBallot));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_clustered_reduce")) {
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformClustered));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_named_barrier")) {
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::NamedBarrier));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_non_uniform_arithmetic")) {
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformArithmetic));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_non_uniform_vote")) {
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniform));
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformVote));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_rotate")) {
    spirvExtensions.push_back("SPV_KHR_subgroup_rotate");
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformRotateKHR));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_shuffle")) {
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformShuffle));
  }

  if (strstr(dev->extensions, "cl_khr_subgroup_shuffle_relative")) {
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupNonUniformShuffleRelative));
  }

  if (strstr(dev->extensions, "cl_khr_work_group_uniform_arithmetic")) {
    spirvExtensions.push_back("SPV_KHR_uniform_group_instructions");
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::GroupUniformArithmeticKHR));
  }

  if (strstr(dev->extensions, "cl_ext_float_atomics")) {
    if (dev->single_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)) {
      spirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::AtomicFloat32AddEXT));
    }
    if (dev->single_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)) {
      spirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::AtomicFloat32MinMaxEXT));
    }
    if (dev->half_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
                                    CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)) {
      spirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::AtomicFloat16AddEXT));
    }
    if (dev->half_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                    CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)) {
      spirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::AtomicFloat16MinMaxEXT));
    }
    if (dev->double_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)) {
      spirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::AtomicFloat64AddEXT));
    }
    if (dev->double_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)) {
      spirvCapabilities.push_back(
          static_cast<cl_uint>(spv::Capability::AtomicFloat64MinMaxEXT));
    }
    if (dev->single_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT) ||
        dev->double_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_ADD_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_ADD_EXT)) {
      spirvExtensions.push_back("SPV_EXT_shader_atomic_float_add");
    }
    if (dev->single_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT) ||
        dev->half_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                    CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT) ||
        dev->double_fp_atomic_caps & (CL_DEVICE_GLOBAL_FP_ATOMIC_MIN_MAX_EXT |
                                      CL_DEVICE_LOCAL_FP_ATOMIC_MIN_MAX_EXT)) {
      spirvExtensions.push_back("SPV_EXT_shader_atomic_float_min_max");
    }
  }

  if (strstr(dev->extensions, "cl_intel_spirv_subgroups")) {
    spirvExtensions.push_back("SPV_INTEL_subgroups");
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SubgroupBufferBlockIOINTEL));
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SubgroupImageBlockIOINTEL));
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SubgroupShuffleINTEL));
  }

  if (strstr(dev->extensions, "cl_intel_split_work_group_barrier")) {
    spirvExtensions.push_back("SPV_INTEL_split_barrier");
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SplitBarrierINTEL));
  }

  if (strstr(dev->extensions, "cl_intel_subgroup_buffer_prefetch")) {
    spirvExtensions.push_back("SPV_INTEL_subgroup_buffer_prefetch");
    spirvCapabilities.push_back(
        static_cast<cl_uint>(spv::Capability::SubgroupBufferPrefetchINTEL));
  }

  if (!spirvExtendedInstructionSets.empty()) {
    dev->num_spirv_extended_instruction_sets =
        spirvExtendedInstructionSets.size();
    dev->spirv_extended_instruction_sets = (const char **)malloc(
        spirvExtendedInstructionSets.size() * sizeof(const char *));
    memcpy(dev->spirv_extended_instruction_sets,
           spirvExtendedInstructionSets.data(),
           spirvExtendedInstructionSets.size() * sizeof(const char *));
  }

  if (!spirvExtensions.empty()) {
    dev->num_spirv_extensions = spirvExtensions.size();
    dev->spirv_extensions =
        (const char **)malloc(spirvExtensions.size() * sizeof(const char *));
    memcpy(dev->spirv_extensions, spirvExtensions.data(),
           spirvExtensions.size() * sizeof(const char *));
  }

  if (!spirvCapabilities.empty()) {
    dev->num_spirv_capabilities = spirvCapabilities.size();
    cl_uint *temp =
        (cl_uint *)malloc(spirvCapabilities.size() * sizeof(cl_uint));
    memcpy(temp, spirvCapabilities.data(),
           spirvCapabilities.size() * sizeof(cl_uint));
    dev->spirv_capabilities = temp;
  }
}

#ifdef __cplusplus
}
#endif
