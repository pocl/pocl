# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/lib/llvmopencl.
# 
# Copyright (c) 2011 Universidad Rey Juan Carlos
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

if USE_LLVM_API
# This library is linked against libpocl to call the kernel compiler
# directly.
noinst_LTLIBRARIES = libllvmpasses.la
endif
# This creates a separate llvmopencl.so that is loaded to 'opt'
# for the pocl-workgroup version.
pkglib_LTLIBRARIES = llvmopencl.la

#the explicit '-DNDEBUG' is a work-around for llvm bug 18253
AM_CXXFLAGS = -I@top_srcdir@/fix-include -I@top_srcdir@/include `@LLVM_CONFIG@ --cxxflags` -fno-rtti
if !LLVM_HAS_ASSERTIONS
AM_CXXFLAGS += -DNDEBUG
endif

AM_LDFLAGS = -module -export-dynamic -version-info ${KERNEL_COMPILER_LIB_VERSION} `@LLVM_CONFIG@ --ldflags`

# When building in API mode, LLVM gets linked into libpocl. Linking
# it in here too gives "symbol redefined" errors with the standalone build.
if !USE_LLVM_API
llvmopencl_la_LIBADD = @LLVM_LIBS@
endif

PASSES_SOURCES = Barrier.h						\
                        BarrierBlock.h BarrierBlock.cc				\
                        Kernel.h Kernel.cc					\
                        ParallelRegion.h ParallelRegion.cc			\
                        CanonicalizeBarriers.h CanonicalizeBarriers.cc		\
                        LoopBarriers.h LoopBarriers.cc				\
                        GenerateHeader.cc Workgroup.h Workgroup.cc		\
						BarrierTailReplication.h BarrierTailReplication.cc	\
                        Flatten.cc IsolateRegions.h IsolateRegions.cc \
						WorkitemReplication.h WorkitemReplication.cc \
						ImplicitLoopBarriers.h ImplicitLoopBarriers.cc \
						WorkItemAliasAnalysis.cc WIVectorize.cc \
						WorkitemHandler.h WorkitemHandler.cc \
						WorkitemLoops.h WorkitemLoops.cc \
						PHIsToAllocas.h PHIsToAllocas.cc \
						BreakConstantGEPs.h BreakConstantGEPs.cpp \
						WorkitemHandlerChooser.h WorkitemHandlerChooser.cc \
						AllocasToEntry.h AllocasToEntry.cc \
						TargetAddressSpaces.h TargetAddressSpaces.cc \
						LLVMUtils.cc LLVMUtils.h \
						VariableUniformityAnalysis.h VariableUniformityAnalysis.cc \
						AutomaticLocals.cc ImplicitConditionalBarriers.cc \
						ImplicitConditionalBarriers.h Scalarizer.cpp \
						DebugHelpers.h DebugHelpers.cc

if USE_LLVM_API
libllvmpasses_la_SOURCES = ${PASSES_SOURCES}
endif
# some scripts remain in the api version, and those require libopencl.so
# so build it even when using the API
llvmopencl_la_SOURCES = ${PASSES_SOURCES}

#llvmopencl_la_LIBADD += @LIBS_LLVMTRANSFORMUTILS@
