bin_PROGRAMS = pocl-llvm-ld
pocl_llvm_ld_SOURCES = llvm-ld.cpp Optimize.cpp
pocl_llvm_ld_LDFLAGS = `@LLVM_CONFIG@ --ldflags` -lLLVM-`@LLVM_CONFIG@ --version`
pocl_llvm_ld_CXXFLAGS = `@LLVM_CONFIG@ --cxxflags`




