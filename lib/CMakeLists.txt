
separate_arguments(HOST_CLANG_FLAGS)
separate_arguments(HOST_LLC_FLAGS)

#add_subdirectory("llvmopencl")

#add_subdirectory("CL")

#############################################################

# must be after cell / tce b/c of LD_FLAGS_BIN
# and after adding lib/CL subdir b/c of libpocl location,
# but before poclu & examples & tests (they need these variables)

# TODO possibly required
set(OPENCL_CFLAGS "") #'$(PTHREAD_CFLAGS)'

if(TESTS_USE_ICD)

  PKG_CHECK_MODULES(OPENCL OpenCL>=1.2)
  if(OPENCL_FOUND)
    set(OPENCL_LIBS ${OPENCL_LIBRARIES})
  else()
    message(WARNING "Tests-with-icd requested, but cannot find OpenCL library via pkg-config")
    set(OPENCL_LIBS "-lOpenCL")
  endif()

  set(OPENCL_EXTLIBS ${OPENCL_LIBS})
  set(OPENCL_CMAKE_OPTIONS "")

else()

  # used by examples / tests

  #OPENCL_LIBS='$(abs_top_builddir)/lib/CL/libpocl.la $(PTHREAD_LIBS) $(LD_FLAGS_BIN)'
  set(OPENCL_LIBS ${LD_FLAGS_BIN} "OpenCL")

  # TODO used by Rodinia makefile.am ?
  #OPENCL_EXTLIBS='-L$(abs_top_builddir)/lib/CL/.libs -lpocl -Xlinker -rpath=${abs_top_builddir}/lib/CL/.libs $(PTHREAD_LIBS) $(LD_FLAGS_BIN)'
  set(OPENCL_EXTLIBS "TODO_FIX_THIS")

  #~ OPENCL_CMAKE=' \
  #~ -D OPENCL_LIBRARIES:FILEPATH=${abs_top_builddir}/lib/CL/.libs/libpocl.so \
  #~ -D OPENCL_LIBRARY:FILEPATH=${abs_top_builddir}/lib/CL/.libs/libpocl.so \
  #~ -D CMAKE_EXE_LINKER_FLAGS:STRING='${CMAKE_THREAD_LIBS_INIT}' \
  #~ -D OPENCL_INCLUDE_DIRS:PATH=${abs_top_srcdir}/include \
  #~ -D OPENCL_INCLUDE_DIR:PATH=${abs_top_srcdir}/include
  #~ '
  set(OPENCL_CMAKE_OPTIONS "-D OPENCL_LIBRARIES:FILEPATH='${LIBPOCL_LOCATION}'
                            -D OPENCL_LIBRARY:FILEPATH='${LIBPOCL_LOCATION}'
                            -D CMAKE_EXE_LINKER_FLAGS:STRING='${CMAKE_THREAD_LIBS_INIT}'
                            -D OPENCL_INCLUDE_DIRS:PATH='${CMAKE_SOURCE_DIR}/include'
                            -D OPENCL_INCLUDE_DIR:PATH='${CMAKE_SOURCE_DIR}/include' ")

endif()

set(OPENCL_LIBS "${OPENCL_LIBS}" PARENT_SCOPE)
set(OPENCL_EXTLIBS "${OPENCL_EXTLIBS}" PARENT_SCOPE)
set(OPENCL_CMAKE_OPTIONS "${OPENCL_CMAKE_OPTIONS}" PARENT_SCOPE)
set(OPENCL_CFLAGS "${OPENCL_CFLAGS}" PARENT_SCOPE)

#add_subdirectory("kernel")

#add_subdirectory("poclu")

