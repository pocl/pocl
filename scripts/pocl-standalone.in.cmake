#!/bin/sh
# pocl-standalone - Run parallelization passes directly on an OpenCL source
#                   file and generate a parallel bytecode and a kernel description
#                   header file.
#
# Copyright (c) 2011-2012 Carlos Sánchez de La Lama / URJC and
#               2011-2013 Pekka Jääskeläinen / TUT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

set -e                          # Abort on errors
if [ -n "$POCL_VERBOSE" ]; then
    set -x
    echo 0=$0 @=$@
fi

target=@OCL_KERNEL_TARGET@

while getopts h:t:o: o
do
    case "$o" in
  h)   header="${OPTARG}";;
  t)   target="${OPTARG}";;
  o)   output_file="${OPTARG}";;
  [?]) echo >&2 "Usage: $0 -o <output_file> <input_file>" && exit 1;;
    esac
done
shift $((${OPTIND}-1))

if [ "x$header" = x ]
then
    echo >&2 "Usage: $0 [-t target] -h <header> -o <output_file> <input_file>" && exit 1
fi

if [ "x$output_file" = x ]
then
    echo >&2 "Usage: $0 [-t target] -h <header> -o <output_file> <input_file>" && exit 1
fi

case $target in
    cellspu-*) target_dir="cellspu";;
    tce*)     target_dir="tce"
               target="tce-tut-llvm"
               ;;
    *)         target_dir="host";;
esac

case $target in
    tce*)     CLANG_FLAGS="@TCE_TARGET_CLANG_FLAGS@"
              LLC_FLAGS="@TCE_TARGET_LLC_FLAGS@"
              LD_FLAGS="@@";;
    cell*)    CLANG_FLAGS="@CELL_TARGET_CLANG_FLAGS@"
              LLC_FLAGS="@CELL_TARGET_LLC_FLAGS@"
              LD_FLAGS="@@";;
    *)        CLANG_FLAGS="@HOST_CLANG_FLAGS@"
              LLC_FLAGS="@HOST_LLC_FLAGS@"
              LD_FLAGS="@HOST_LD_FLAGS@";;
# TODO
#    @TARGET@) CLANG_FLAGS="@TARGET_CLANG_FLAGS@"
#              LLC_FLAGS="@TARGET_LLC_FLAGS@"
#              LD_FLAGS="@TARGET_LD_FLAGS@";;
esac
CLANG_FLAGS="$CLANG_FLAGS -fasm -fsigned-char -Xclang -ffake-address-space-map"
echo $target
# With fp-contract we get calls to fma with processors which do not
# have fma instructions. These ruin the performance. Better to have
# the mul+add separated in the IR.
CLANG_FLAGS="$CLANG_FLAGS -ffp-contract=off"

tempdir="`dirname ${output_file}`/.pocl$$"
mkdir ${tempdir}

kernel_bc="${tempdir}/kernel.bc"

pocl_kernel_compiler_lib=@LLVMOPENCL_LOCATION@
@CLANG@ ${CLANG_FLAGS} $EXTRA_CLANG_FLAGS -c -emit-llvm @ADD_INCLUDE@ -include @KERNEL_INCLUDE_DIR@/_kernel.h -o ${kernel_bc} -x cl $1
rm -f ${header}
@LLVM_OPT@ ${LLC_FLAGS} -load=$pocl_kernel_compiler_lib -generate-header -disable-output -header=${header} ${kernel_bc}

linked_bc="${tempdir}/linked.bc"
linked_out="${linked_bc}.out"
full_target_dir=@FULL_TARGET_DIR@

@LLVM_LINK@ -o ${linked_bc} ${kernel_bc} $full_target_dir/kernel-$target.bc

OPT_SWITCH="-O3"

if test "x$POCL_KERNEL_COMPILER_OPT_SWITCH" != "x";
then
OPT_SWITCH=$POCL_KERNEL_COMPILER_OPT_SWITCH
fi

# -disable-simplify-libcalls was added because of TCE (it doesn't have
# a runtime linker which could link the libs later on), but it might
# also be harmful for wg-vectorization where we want to try to vectorize
# the code it wants to convert e.g. to a memset or a memcpy

@LLVM_OPT@ ${LLC_FLAGS} \
    -load=${pocl_kernel_compiler_lib} -domtree -workitem-handler-chooser -break-constgeps -generate-header -flatten -always-inline \
    -globaldce -simplifycfg -loop-simplify -uniformity -phistoallocas -isolate-regions -implicit-loop-barriers -implicit-cond-barriers \
    -loop-barriers -barriertails -barriers -isolate-regions -add-wi-metadata -wi-aa -workitemrepl -workitemloops \
    -allocastoentry -workgroup -kernel=${kernel} -disable-simplify-libcalls \
    -target-address-spaces \
    ${EXTRA_OPTS} ${OPT_SWITCH} -instcombine -header=/dev/null ${FP_CONTRACT} -o ${output_file} ${linked_bc}


rm -fr ${tempdir}
