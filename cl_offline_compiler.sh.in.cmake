#!/bin/bash

#=============================================================================
#   cl_offline_compiler.sh script
#
#   Copyright (c) 2017 Michal Babej / Intel Finland Oy
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

# Takes an OpenCL C source file and uses Clang + LLVM-SPIRV to convert it to
# a SPIR-V binary. Used by OpenCL-CTS in offline compilation mode. Unlike
# poclcc, this does not return a poclbinary, but a SPIR-V binary, and is
# somewhat independent of PoCL (there is some option / extension processing
# that is PoCL-specific)

# Likely requires recent Clang (14 tested, 13 and older untested).
# Might work with -cl-std < 3.0 with older Clangs, but 3.0 requires 14+

# mandatory arguments:
#     --source FILE          OpenCL C source file to compile
#     --output FILE          SPIR-V or binary file to create
#     --cl-device-info FILE  OpenCL device info file
#
# optional arguments:
#     --debug                Enable some debugging output
#     --mode                 compilation mode (spir-v or binary)

MODE=spir-v
DEBUG=false

for i in "$@"; do
  case $i in
    --source=*)
      SOURCE="${i#*=}"
      shift
      ;;
    --output=*)
      OUTPUT="${i#*=}"
      shift
      ;;
    --cl-device-info=*)
      CL_DEV_INFO="${i#*=}"
      shift
      ;;
    --mode=*)
      MODE="${i#*=}"
      shift
      ;;
    --debug)
      DEBUG=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      ;;
  esac
done

if [ -z "$SOURCE" ] || [ -z "$OUTPUT" ] || [ -z "$CL_DEV_INFO" ]; then
  echo "USAGE: $0 --source=/path/source.cl --output=/path/output.spv --cl-device-info=/path/to/dev-info.txt [--debug] [--mode=MODE] -- [opencl compile options...]"
  exit 1
fi

if [ "$MODE" != "spir-v" ]; then
  echo "this script only compiles to SPIR-V"
  exit 1
fi

CL_IS_30=false
CL_FAST_MATH=false
CL_UNSAFE_MATH=false
BUILD_OPTIONS=""
CL_STD="-cl-std=CL1.2"

for i in "$@"; do
  case $i in
    -cl-std=*)
      CL_STD="$i"
      shift
      ;;
    -cl-fast-relaxed-math)
      CL_FAST_MATH=true
      shift
      ;;
    -cl-unsafe-math)
      CL_UNSAFE_MATH=true
      shift
      ;;
    *)
      BUILD_OPTIONS="$BUILD_OPTIONS $i"
      shift
      ;;
  esac
done

if [ "$CL_STD" = "-cl-std=CL3.0" ]; then
  CL_IS_30=true
fi


TARGET=none
DEV_VER=100
DEV_C_VER=100
CL_EXT_DEFS="-D__ENDIAN_LITTLE__=1"
# TODO __opencl_c_int64 && atomics might not be supported by all PoCL devices
CL_EXTS="-Xclang -cl-ext=-all"

if [ -e "${CL_DEV_INFO}" ]; then

  echo "CL_DEV_INFO: ${CL_DEV_INFO}"
  source ${CL_DEV_INFO}

  if [ "$CL_DEVICE_IMAGE_SUPPORT" -eq 1 ]; then
    if [ "$CL_IS_30" = "true" ]; then
      CL_EXTS="${CL_EXTS},+__opencl_c_images"
    fi
    CL_EXT_DEFS="${CL_EXT_DEFS} -D__IMAGE_SUPPORT__=1"
  fi

  if [ "$CL_FAST_MATH" = "true" ]; then
    CL_EXT_DEFS="${CL_EXT_DEFS} -cl-finite-math-only"
    CL_UNSAFE_MATH=true
  fi

  if [ "$CL_UNSAFE_MATH" = "true" ]; then
    CL_EXT_DEFS="${CL_EXT_DEFS} -cl-no-signed-zeros -cl-mad-enable -ffp-contract=fast"
  fi

  if [[ "$CL_DEVICE_VERSION" =~ "PoCL" ]] && [ "$CL_IS_30" = "true" ]; then
    if [[ "$CL_DEVICE_VERSION" =~ "basic" ]] || [[ "$CL_DEVICE_VERSION" =~ "pthread" ]]; then
      CL_EXT_DEFS="${CL_EXT_DEFS} -D__opencl_c_named_address_space_builtins=1 -D__opencl_c_int64=1 -D__opencl_c_atomic_order_acq_rel=1 -D__opencl_c_atomic_order_seq_cst=1 -D__opencl_c_atomic_scope_device=1 -D__opencl_c_program_scope_global_variables=1 "
      CL_EXTS="${CL_EXTS},+__opencl_c_named_address_space_builtins,+__opencl_c_int64,+__opencl_c_atomic_order_acq_rel,+__opencl_c_atomic_order_seq_cst,+__opencl_c_atomic_scope_device,+__opencl_c_program_scope_global_variables"
    fi
  fi

  if [[ $CL_DEVICE_ADDRESS_BITS==64 ]]; then
    TARGET=spir64-unknown-unknown
  elif [[ $CL_DEVICE_ADDRESS_BITS==32 ]]; then
    TARGET=spir-unknown-unknown
  else
    echo "unknown device address bits: ${CL_DEVICE_ADDRESS_BITS}"
    exit 1
  fi

  if [[ "$CL_DEVICE_VERSION" =~ "OpenCL 3.0" ]]; then
    DEV_VER=300
  elif [[ "$CL_DEVICE_VERSION" =~ "OpenCL 2.2" ]]; then
    DEV_VER=220
  elif [[ "$CL_DEVICE_VERSION" =~ "OpenCL 2.1" ]]; then
    DEV_VER=210
  elif [[ "$CL_DEVICE_VERSION" =~ "OpenCL 2.0" ]]; then
    DEV_VER=200
  elif [[ "$CL_DEVICE_VERSION" =~ "OpenCL 1.2" ]]; then
    DEV_VER=120
  else
    echo "unknown device version: ${CL_DEVICE_VERSION}"
    exit 1
  fi
  BUILD_OPTIONS="$BUILD_OPTIONS -D__OPENCL_VERSION__=${DEV_VER}"

  if [[ "$CL_STD" =~ "CL3.0" ]]; then
    DEV_C_VER=300
  elif [[ "$CL_STD" =~ "CL2.2" ]]; then
    DEV_C_VER=220
  elif [[ "$CL_STD" =~ "CL2.1" ]]; then
    DEV_C_VER=210
  elif [[ "$CL_STD" =~ "CL2.0" ]]; then
    DEV_C_VER=200
  elif [[ "$CL_STD" =~ "CL1.2" ]]; then
    DEV_C_VER=120
  else
    echo "unknown device C version: ${CL_STD}"
    exit 1
  fi
  BUILD_OPTIONS="$BUILD_OPTIONS -D__OPENCL_C_VERSION__=${DEV_C_VER}"


  for EXT in $CL_DEVICE_EXTENSIONS ; do
    CL_EXT_DEFS="${CL_EXT_DEFS} -D${EXT}"
    CL_EXTS="${CL_EXTS},+${EXT}"
    if [ "$CL_IS_30" = "true" ]; then
      case $EXT in
        cl_khr_3d_image_writes)
          CL_EXT_DEFS="${CL_EXT_DEFS} -D__opencl_c_3d_image_writes=1"
          CL_EXTS="${CL_EXTS},+__opencl_c_3d_image_writes"
          ;;
        cl_khr_fp64)
          CL_EXT_DEFS="${CL_EXT_DEFS} -D__opencl_c_fp64=1"
          CL_EXTS="${CL_EXTS},+__opencl_c_fp64"
          ;;
        cl_khr_fp16)
          CL_EXT_DEFS="${CL_EXT_DEFS} -D__opencl_c_fp16=1"
          CL_EXTS="${CL_EXTS},+__opencl_c_fp16"
          ;;
      esac
    fi
  done
  BUILD_OPTIONS="${BUILD_OPTIONS} ${CL_EXT_DEFS} ${CL_EXTS}"

  # TODO check IL_VER

fi

if [ "$DEBUG" = "true" ]; then
  echo "SOURCE: ${SOURCE}"
  echo "OUTPUT: ${OUTPUT}"
fi

ALL_OPTIONS="--target=${TARGET} -x cl ${CL_STD} ${BUILD_OPTIONS} -o ${SOURCE}.bc -emit-llvm -c ${SOURCE}"

LLVM_SPIRV_OPTIONS="--spirv-gen-kernel-arg-name-md --spirv-max-version=1.2 -o ${OUTPUT} ${SOURCE}.bc"

if [ "$DEBUG" = "true" ]; then
  echo "Running @CLANG@ ${ALL_OPTIONS}"
fi
@CLANG@ ${ALL_OPTIONS} || exit 1

if [ "$DEBUG" = "true" ]; then
  echo "Running @LLVM_SPIRV@ ${LLVM_SPIRV_OPTIONS}"
fi
@LLVM_SPIRV@ ${LLVM_SPIRV_OPTIONS} || exit 1

exit 0
