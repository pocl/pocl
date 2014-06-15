
if(UNIX)
  # build-dir script
  get_target_property(LLVMOPENCL_LOCATION "llvmopencl" LOCATION)
  set(KERNEL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include")
  set(ADD_INCLUDE "-I${CMAKE_BINARY_DIR}")
  set(FULL_TARGET_DIR "${CMAKE_BINARY_DIR}/lib/kernel/\${target_dir}")
  configure_file("pocl-standalone.in.cmake" "pocl-standalone" ESCAPE_QUOTES @ONLY)

  # install script
  set(LLVMOPENCL_LOCATION "${POCL_INSTALL_PRIVATE_LIBDIR}/llvmopencl.so")
  set(KERNEL_INCLUDE_DIR "${POCL_INSTALL_PRIVATE_HEADER_DIR}")
  set(ADD_INCLUDE "")
  set(FULL_TARGET_DIR "${POCL_INSTALL_PRIVATE_DATADIR}")
  configure_file("pocl-standalone.in.cmake" "pocl-standalone.install" ESCAPE_QUOTES @ONLY)

  install(FILES "${CMAKE_CURRENT_BINARY_DIR}/pocl-standalone.install" RENAME "pocl-standalone"
        DESTINATION ${POCL_INSTALL_PUBLIC_BINDIR})
endif()
