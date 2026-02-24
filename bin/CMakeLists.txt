#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2016 pocl developers
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

set_opencl_header_includes()

if(ENABLE_SPIRV)
  # TODO refactor spirv-parser into its own library
  set(SPVSRC "${CMAKE_SOURCE_DIR}/lib/CL/devices")
  if(ENABLE_ICD)
    add_executable(spirv_fix_atomic_compare_exchange spirv_fix_atomic_compare_exchange.cc
      "${SPVSRC}/spirv_parser.hh" "${SPVSRC}/spirv_parser.cc")
    # Avoid POCL_EXPORT here
    target_compile_definitions(spirv_fix_atomic_compare_exchange PRIVATE SPIRV_PARSER_LIB)
  else()
    add_executable(spirv_fix_atomic_compare_exchange spirv_fix_atomic_compare_exchange.cc)
  endif()
  harden(spirv_fix_atomic_compare_exchange)
  target_link_libraries(spirv_fix_atomic_compare_exchange poclu ${OPENCL_LIBS})
  target_include_directories(spirv_fix_atomic_compare_exchange PRIVATE "${SPVSRC}")
  # Installed executables need rpath to find the PoCL library.
  # CMAKE_INSTALL_RPATH_USE_LINK_PATH handles external dependencies like LLVM,
  # but the PoCL library is linked from the build tree, so its install
  # location must be added explicitly.
  if(NOT STATIC_LLVM)
    set_target_properties(spirv_fix_atomic_compare_exchange PROPERTIES
      INSTALL_RPATH "${POCL_INSTALL_PUBLIC_LIBDIR}")
  endif()
endif()

if(ENABLE_POCLCC)
  add_executable(poclcc poclcc.c)
  harden(poclcc)
  target_link_libraries(poclcc poclu ${OPENCL_LIBS})
  # Installed executables need rpath to find the PoCL library.
  # CMAKE_INSTALL_RPATH_USE_LINK_PATH handles external dependencies like LLVM,
  # but the PoCL library is linked from the build tree, so its install
  # location must be added explicitly.
  if(NOT STATIC_LLVM)
    set_target_properties(poclcc PROPERTIES
      INSTALL_RPATH "${POCL_INSTALL_PUBLIC_LIBDIR}")
  endif()

  install(TARGETS "poclcc" RUNTIME
          DESTINATION "${POCL_INSTALL_PUBLIC_BINDIR_REL}" COMPONENT "poclcc")

  set("CPACK_DEBIAN_POCLCC_PACKAGE_NAME" "poclcc")
  list(APPEND CPACK_DEBIAN_POCLCC_PACKAGE_DEPENDS "opencl-icd")

  add_symlink_to_built_opencl_dynlib(poclcc)
endif()

if(ENABLE_MLIR)
  add_subdirectory("pocl-mlir-opt")
endif()

pass_through_cpack_vars()
