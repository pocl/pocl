#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2024 Michal Babej / Intel Finland Oy
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#   THE SOFTWARE.
#
#=============================================================================

set(TS_NAME "UnitCL")
set(TS_BASEDIR "${TESTSUITE_BASEDIR}/${TS_NAME}")
set(TS_BUILDDIR "${TS_BASEDIR}/src/${TS_NAME}-build")
set(TS_SRCDIR "${TS_BASEDIR}/src/${TS_NAME}")

if(NOT HAVE_GIT)
  message(STATUS "Disabling testsuite ${TS_NAME}, requires git to checkout sources")
  return()
endif()

if(ENABLE_ICD AND (NOT OPENCL_LIBRARIES))
  message(STATUS "Disabling ${TS_NAME}, can't find OpenCL library")
  return()
endif()

# for SPIR-V tests
find_program(SPIRV_AS "spirv-as")
find_program(SPIRV_VAL "spirv-val")
if(SPIRV_AS AND SPIRV_VAL)
  set(HAVE_SPIRV_TOOLS 1)
endif()

# if ENABLE_CONFORMANCE is OFF (default), the UnitCL reports a lot of failures with CPU driver
if(ENABLE_HOST_CPU_DEVICES AND (NOT ENABLE_CONFORMANCE))
  message(WARNING "UnitCL + CPU driver need ENABLE_CONFORMANCE=ON, disabling CTS")
  return()
endif()

message(STATUS "Enabling testsuite ${TS_NAME}")
list(APPEND ACTUALLY_ENABLED_TESTSUITES "${TS_NAME}")
set(ACTUALLY_ENABLED_TESTSUITES ${ACTUALLY_ENABLED_TESTSUITES} PARENT_SCOPE)


set(UPSTREAM_CMAKE_ARGS "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}")
if(CMAKE_C_COMPILER_LAUNCHER)
  list(APPEND UPSTREAM_CMAKE_ARGS "-DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}")
endif()
if(CMAKE_CXX_COMPILER_LAUNCHER)
  list(APPEND UPSTREAM_CMAKE_ARGS "-DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}")
endif()

ExternalProject_Add(
  ${TS_NAME}
  PREFIX "${TS_BASEDIR}"
  GIT_REPOSITORY "https://github.com/uxlfoundation/oneapi-construction-kit.git"
  GIT_TAG "main"
  ${GIT_OPTIONS}
  CONFIGURE_COMMAND ${CMAKE_COMMAND}
    ${UPSTREAM_CMAKE_ARGS}
    -DCA_ENABLE_DOCUMENTATION=0 # disable docs
    -DCA_ENABLE_API=cl          # disable Vulkan
    -DCA_RUNTIME_COMPILER_ENABLED=0 # disable runtime compiler
    -DCA_CL_ENABLE_OFFLINE_KERNEL_TESTS=0 # disable offline tests (needs CLC or runtime compiler)
    "${TS_BASEDIR}/src/${TS_NAME}"
  BUILD_COMMAND ${CMAKE_COMMAND} --build ${TS_BUILDDIR}/source/cl/test/UnitCL
  INSTALL_COMMAND ""
)

set_target_properties(${TS_NAME} PROPERTIES EXCLUDE_FROM_ALL TRUE)
add_dependencies(prepare_examples ${TS_NAME})

#***************************************************************************

set(TEST_NAMES
Arm64KillerTest
BaseExecution
BitSelectTest
BuildOptionsTest
BuildProgramQuery/clGetProgramBuildInfoTestScalarQueryOpenCL30
CheckAPI/clEnqueueFillImageTest
CheckMacros/FeatureMacroTest
cl3DImageWriteExtensionTest
clBuild
clCloneKernelRunTest
clCloneKernelTest
clCompile
clCreate
clEnqueue
clFinishTest
clFlushTest
clGet
clIcdGetPlatformIDsKHRTest
cl_intel_required_subgroup_size_Test
clLink
clRelease
clRetain
clSet
clSVM
clUnloadCompilerTest
clUnloadPlatformCompilerTest
clWaitForEventsTest
CompileOptionsTest
ContextTest
Default
DeviceQuery
Dimensions
EdgeCase
EndianMacros
EventProfilingQuery
ExampleUnitCLTest_Host
Execution/*-Execution/Half*:Execution/WorkGroupCollective
Execution/Half
Execution/WorkGroupCollective
FastMath/FastMathTest
FLOAT/clEnqueueCopyImageParamTest
FLOAT/clEnqueueWriteImageVerify
FLOAT/clGetImageInfoParamTest
FuzzTest
GetEnqueuedLocalSizeTest
HALF_FLOAT/clEnqueueCopyImageParamTest
HALF_FLOAT/clEnqueueWriteImageVerify
HALF_FLOAT/clGetImageInfoParamTest
host_clGetDeviceInfoTest
ImageMacros/ImageMacrosTest
InvalidFlags/clGetSupportedImageFormatsFlagsTest
InvalidKernels/printfBuiltinInvalidTest
KernelArgumentTypesTest
KernelExecInfoCodeplayTests/KernelExecInfoCodeplayUSMFlags
KernelExecInfoCodeplayUSMPtrs
KernelFeaturesTest
Limits/StackSizeTest
Macros/MacrosTest
MemObjectQuery/clGetMemObjectInfoUsesSVMPointerTest
MemObjTypeTest/clEnqueueMapImageTests
OneArgRelational
PlatformQuery/clGetPlatformInfoTestOpenCL30
ProgramQuery/clGetProgramInfoTestParam
SelectTest
SIGNED
SingleIntger/CtzTest
SNORM
SoftMath/SoftMathTestNativeFloat
SoftMath/SoftMathTestOneArgInteger
SoftMath/SoftMathTestTwoArgInteger
SubGroupsUnsupportedTest
TwoArgRelational
UNORM
UNSIGNED
USM
ValidKernels/printfBuiltinValidTest
VariousNDRangeValues/clEnqueueNDRangeKernelWorkItemTest
ZeroDimensions/clEnqueueNDRangeKernelZeroDimension
)

foreach(TEST_NAME IN TEST_NAMES)
  string(REPLACE "/" "_" NORM_TEST_NAME "${TS_NAME}_${TEST_NAME}")
add_test(NAME "${NORM_TEST_NAME}"
         COMMAND "${TS_BUILDDIR}/bin/UnitCL" "--gtest-filter=${TEST_NAME}*"
set_property(TEST "${NORM_TEST_NAME}"
         PROPERTY LABELS ${TS_NAME})
set_property(TEST "${NORM_TEST_NAME}"
         PROPERTY ENVIRONMENT "POCL_MEMORY_LIMIT=8")
endforeach()

#***************************************************************************
