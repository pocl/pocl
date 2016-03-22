# Process this file with automake to produce Makefile.in (in this,
# and all subdirectories).
# Makefile.am for pocl/lib/llvmopencl.
# 
# Copyright (c) 2011 Universidad Rey Juan Carlos
#               2011-2015 Pekka Jääskeläinen 
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

# This library is linked against libpocl to call the kernel compiler
# passes directly from the host library.
# This library also contains compiler driver helper functions,
# not linked to llvmopencl.so
noinst_LTLIBRARIES = libllvmpasses.la

# This creates a separate llvmopencl.so that is loaded to 'opt'
# for the pocl-workgroup version.
pkglib_LTLIBRARIES = llvmopencl.la

AM_CXXFLAGS = -I@top_srcdir@/fix-include -I@top_srcdir@/include \
              -I$(top_srcdir)/lib/CL -I@top_srcdir@/lib/CL/devices \
	           @LLVM_CXX_FLAGS@

AM_LDFLAGS = -module -export-dynamic @LLVM_LDFLAGS@ -version-info ${KERNEL_COMPILER_LIB_VERSION}

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
						WorkItemAliasAnalysis.cc \
						WorkitemHandler.h WorkitemHandler.cc \
						WorkitemLoops.h WorkitemLoops.cc \
						PHIsToAllocas.h PHIsToAllocas.cc \
						BreakConstantGEPs.h BreakConstantGEPs.cpp \
						WorkitemHandlerChooser.h WorkitemHandlerChooser.cc \
						AllocasToEntry.h AllocasToEntry.cc \
						TargetAddressSpaces.h TargetAddressSpaces.cc \
						LLVMUtils.cc LLVMUtils.h LLVMFileUtils.cc \
						VariableUniformityAnalysis.h VariableUniformityAnalysis.cc \
						AutomaticLocals.cc ImplicitConditionalBarriers.cc \
						ImplicitConditionalBarriers.h \
						DebugHelpers.h DebugHelpers.cc \
						linker.h CompilerWarnings.h

#add compiler driver sources
libllvmpasses_la_SOURCES = ${PASSES_SOURCES} linker.cpp

# The offline compiler script (pocl-standalone) calls the kernel compiler passes 
# from the command line, those require libllvmopencl.so.
llvmopencl_la_SOURCES = ${PASSES_SOURCES}

EXTRA_DIST = CMakeLists.txt
