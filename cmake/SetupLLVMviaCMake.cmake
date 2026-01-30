
#=============================================================================
#   CMake build system files for detecting Clang and LLVM
#
#   Copyright (c) 2014-2020 pocl developers
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

# LLVM_CMAKE_DIR contains LLVM_INSTALL_PREFIX/lib/cmake/llvm
get_filename_component(CLANG_CMAKE_DIR "${LLVM_CMAKE_DIR}" DIRECTORY)
# CLANG_CMAKE_DIR will contain LLVM_INSTALL_PREFIX/lib/cmake/clang
set(LLD_CMAKE_DIR "${CLANG_CMAKE_DIR}/lld")
set(CLANG_CMAKE_DIR "${CLANG_CMAKE_DIR}/clang")

message(STATUS "LLVM CMAKE dir: ${LLVM_CMAKE_DIR}")
message(STATUS "CLANG CMAKE dir: ${CLANG_CMAKE_DIR}")
message(STATUS "LLD CMAKE dir: ${LLD_CMAKE_DIR}")

find_package(Clang CONFIG REQUIRED HINTS "${CLANG_CMAKE_DIR}" NO_DEFAULT_PATH)
list(APPEND CMAKE_MODULE_PATH "${LLD_CMAKE_DIR}" "${CLANG_CMAKE_DIR}" "${LLVM_CMAKE_DIR}" )
list(REMOVE_DUPLICATES CMAKE_MODULE_PATH)

message(STATUS "Using CMake module path: ${CMAKE_MODULE_PATH}")

include(AddLLVM)
include(AddClang)

# check the host compiler is good enough to compile LLVM
include(CheckCompilerVersion)
# check if atomics are supported with/without extra LD flags
include(CheckAtomic)

set(LLVM_BINARY_SUFFIX "-${LLVM_VERSION_MAJOR}")
set(LLVM_BINDIR "${LLVM_TOOLS_BINARY_DIR}")
set(LLVM_LIBDIR "${LLVM_LIBRARY_DIR}")

#message(STATUS "LLVM BINDIR: ${LLVM_BINDIR} LLVM_LIBDIR: ${LLVM_LIBDIR}")

set(LLVM_PREFIX_CMAKE "${LLVM_INSTALL_PREFIX}")
#run_llvm_config(LLVM_PREFIX --prefix)
# on windows, llvm-config returs "C:\llvm_prefix/bin" mixed style paths,
# and cmake doesn't like the "\" - thinks its an escape char..
#file(TO_CMAKE_PATH "${LLVM_PREFIX}" LLVM_PREFIX_CMAKE)

set(LLVM_VERSION_FULL ${LLVM_PACKAGE_VERSION})

set(LLVM_ALL_TARGETS ${LLVM_TARGETS_TO_BUILD})
set(LLVM_HOST_TARGET ${LLVM_HOST_TRIPLE})
set(LLVM_BUILD_MODE ${LLVM_BUILD_TYPE})
set(LLVM_ASSERTS_BUILD ${LLVM_ENABLE_ASSERTIONS})

set(LLVM_HAS_RTTI "${LLVM_ENABLE_RTTI}")

set(LLC_TRIPLE ${LLVM_TARGET_TRIPLE})
if(NOT LLC_TRIPLE)
  # fallback for older LLVM
  set(LLC_TRIPLE ${TARGET_TRIPLE})
endif()
if(NOT LLC_TRIPLE)
  message(FATAL_ERROR "LLC triple unset: ${LLVM_TARGET_TRIPLE}")
endif()

###########################################################################

# TODO llvm_add_library() calls llvm_update_compile_flags()
# which sets the flags directly on the target.
# these are unsolved (LLVM_DEFINITIONS only contains -D defs)
#string(REPLACE " -pedantic" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")
#string(REGEX REPLACE "-W[^ ]*" "" LLVM_CXXFLAGS "${LLVM_CXXFLAGS}")

# TODO LDFLAGS are probably set on target
#if(MSVC)
#  string(REPLACE "-L${LLVM_LIBDIR}" "" LLVM_LDFLAGS "${LLVM_LDFLAGS}")
#  string(STRIP "${LLVM_LDFLAGS}" LLVM_LDFLAGS)
#  file(TO_CMAKE_PATH "${LLVM_LDFLAGS}" LLVM_LDFLAGS)
#endif()

if(LLVM_BUILD_MODE MATCHES "Debug")
  set(LLVM_BUILD_MODE_DEBUG 1)
else()
  set(LLVM_BUILD_MODE_DEBUG 0)
endif()

#if (WIN32)
# set(LLVM_HOST_TARGET "x86_64-pc")
# ...

####################################################################

set(POCL_CLANG_COMPONENTS clangSupport clangCodeGen clangFrontendTool
  clangFrontend clangAPINotes clangDriver clangSerialization
  clangParse clangSema clangRewrite clangRewriteFrontend
  clangStaticAnalyzerFrontend clangStaticAnalyzerCheckers
  clangStaticAnalyzerCore clangAnalysis clangEdit
  clangAST clangASTMatchers clangLex clangBasic)

set(POCL_LLVM_COMPONENTS
  LLVMDemangle
  LLVMSupport
  LLVMCore
  LLVMCodeGen
  LLVMCodeGenTypes
  LLVMCoverage
  LLVMIRPrinter
  LLVMIRReader
  LLVMBitReader
  LLVMBitWriter
  LLVMBitstreamReader
  LLVMGlobalISel
  LLVMBinaryFormat
  LLVMTransformUtils
  LLVMInstrumentation
  LLVMInstCombine
  LLVMScalarOpts
  LLVMFrontendDriver
  LLVMFrontendHLSL
  LLVMipo
  LLVMVectorize
  LLVMLinker
  LLVMAnalysis
  LLVMLTO
  LLVMMC
  LLVMMCParser
  LLVMObjCopy
  LLVMObject
  LLVMOption
  LLVMRemarks
  LLVMDebugInfoDWARF
  LLVMExecutionEngine
  LLVMTarget
  LLVMPasses
  LLVMTargetParser
  LLVMLibDriver
  LLVMWindowsDriver
)

if("X86" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND POCL_LLVM_COMPONENTS
    LLVMX86CodeGen
    LLVMX86AsmParser
    LLVMX86Disassembler
    LLVMX86TargetMCA
    LLVMX86Desc
    LLVMX86Info)
endif()

if("RISCV" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND POCL_LLVM_COMPONENTS
    LLVMRISCVCodeGen
    LLVMRISCVAsmParser
    LLVMRISCVDisassembler
    LLVMRISCVDesc
    LLVMRISCVTargetMCA
    LLVMRISCVInfo)
endif()

if("AArch64" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND POCL_LLVM_COMPONENTS
    LLVMAArch64CodeGen
    LLVMAArch64AsmParser
    LLVMAArch64Disassembler
    LLVMAArch64Desc
    LLVMAArch64Info
    LLVMAArch64Utils)
endif()

if("ARM" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND POCL_LLVM_COMPONENTS
    LLVMARMCodeGen
    LLVMARMAsmParser
    LLVMARMDisassembler
    LLVMARMDesc
    LLVMARMInfo
    LLVMARMUtils)
endif()

if("SPIRV" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND POCL_LLVM_COMPONENTS
    LLVMSPIRVCodeGen
    LLVMSPIRVDesc
    LLVMSPIRVInfo
    LLVMSPIRVAnalysis)
endif()

if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND POCL_LLVM_COMPONENTS
    LLVMNVPTXCodeGen
    LLVMNVPTXDesc
    LLVMNVPTXInfo)
endif()

# LLVM_ENABLE_SHARED_LIBS = LLVM is built with shared component libraries
# (libLLVMxyz.so ); the same applies to libclangxyz.so

if(STATIC_LLVM)
  # static link LLVM
  if(LLVM_ENABLE_SHARED_LIBS)
    message(FATAL_ERROR "STATIC_LLVM=ON but LLVM was built with shared libs (BUILD_SHARED_LIBS)")
  endif()
  # libLLVM & libclang-cpp are always shared (AFAIK)
  set(LLVM_LIBS "${POCL_LLVM_COMPONENTS}")
  set(CLANG_LIBS "${POCL_CLANG_COMPONENTS}")
  set(LLVM_LINK_TYPE STATIC)
  # these are enabled when LLVM is built with -DLLVM_BUILD_LLVM_DYLIB,
  # but we must disable these if we're linking to static component libraries
  set(CLANG_LINK_CLANG_DYLIB OFF)
  set(LLVM_LINK_LLVM_DYLIB OFF)
  set(DISABLE_LLVM_LINK_LLVM_DYLIB ON)
else()
  # shared link LLVM
  # check if we have shared components
  if(LLVM_ENABLE_SHARED_LIBS)
    set(LLVM_LIBS "${POCL_LLVM_COMPONENTS}")
    set(CLANG_LIBS "${POCL_CLANG_COMPONENTS}")
  else()
    # if shared component libs are disabled, link with DYLIB (libLLVM.so)
    if(NOT "LLVM" IN_LIST LLVM_AVAILABLE_LIBS)
      message(FATAL_ERRROR "Can't find LLVM in LLVM_AVAILABLE_LIBS")
    endif()
    if(NOT "clang-cpp" IN_LIST CLANG_EXPORTED_TARGETS)
      message(FATAL_ERRROR "Can't find clang-cpp in CLANG_EXPORTED_TARGETS")
    endif()
    set(LLVM_LIBS "LLVM")
    set(CLANG_LIBS "clang-cpp")
  endif()
  set(LLVM_LINK_TYPE SHARED)
endif()

# if enabled, CPU driver on Windows will use lld-link (invoked via library API)
# to link final kernel object files, instead of the default Clang driver linking.
set(CPU_USE_LLD_LINK_WIN32 OFF)
# TODO does not yet work with MINGW; tested but the linked DLL is empty
if(ENABLE_HOST_CPU_DEVICES AND ENABLE_LLVM AND STATIC_LLVM AND MSVC)
  find_package(LLD ${LLVM_VERSION} EXACT CONFIG HINTS "${LLD_CMAKE_DIR}" NO_DEFAULT_PATH)
  if(lldCommon IN_LIST LLD_EXPORTED_TARGETS)
    message(STATUS "Using lld-link via library to link kernels for CPU devices")
    set(CPU_USE_LLD_LINK_WIN32 ON)
    list(APPEND LLVM_LIBS lldCommon lldCOFF lldMinGW)
  endif()
endif()

set(POCL_CLANG_LINK_TARGETS ${CLANG_LIBS})
message(STATUS "POCL_CLANG_LINK_TARGETS: ${POCL_CLANG_LINK_TARGETS}")
set(POCL_LLVM_LINK_TARGETS ${LLVM_LIBS})
message(STATUS "POCL_LLVM_LINK_TARGETS: ${POCL_LLVM_LINK_TARGETS}")

####################################################################

# to avoid creating an install target in add_clang_library()
set(LLVM_INSTALL_TOOLCHAIN_ONLY ON)

if(LLVM_ENABLE_SHARED_LIBS OR STATIC_LLVM)
  # neccesary hack. Unfortunately CMake targets for Clang components have each
  # hardcoded "LLVM" Target in their INTERFACE_LINK_LIBRARIES, which is a problem
  # if we want to link against Clang and LLVM components instead of libLLVM
  # (e.g. for STATIC_LLVM=ON). Manually remove the LLVM dependency.
  # POCL_CLANG_LINK_TARGETS has indirect dependencies -> remove
  # LLVM from all Clang Targets (CLANG_EXPORTED_TARGETS)
  foreach(CLANG_TARGET IN LISTS CLANG_EXPORTED_TARGETS)
    if(TARGET ${CLANG_TARGET})
      #message(STATUS "removing LLVM dependency on ${CLANG_TARGET}")
      unset(IFACE_LIBS)
      get_target_property(IFACE_LIBS ${CLANG_TARGET} INTERFACE_LINK_LIBRARIES)
      if(IFACE_LIBS)
        list(REMOVE_ITEM IFACE_LIBS "LLVM")
        #message(STATUS "New interface link libraries: ${IFACE_LIBS}")
        set_target_properties(${CLANG_TARGET} PROPERTIES INTERFACE_LINK_LIBRARIES "${IFACE_LIBS}")
      #else()
        #message(STATUS "INTERFACE_LINK_LIBRARIES for ${CLANG_TARGET} empty, skipping")
      endif()
    else()
      #message(STATUS "${CLANG_TARGET} is not a valid target, skipping")
    endif()
  endforeach()
endif()
