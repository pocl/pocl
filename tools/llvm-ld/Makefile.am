pkglibexec_PROGRAMS = pocl-llvm-ld
pocl_llvm_ld_SOURCES = llvm-ld.cpp Optimize.cpp
pocl_llvm_ld_LDFLAGS = `@LLVM_CONFIG@ --ldflags`
pocl_llvm_ld_LDADD = -lLLVM-${LLVM_VERSION}
pocl_llvm_ld_CXXFLAGS = `@LLVM_CONFIG@ --cxxflags`




