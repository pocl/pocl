
#cl.hpp: $(top_srcdir)/include/CL/cl.hpp.in $(top_srcdir)/tools/patches/khronos_cl.hpp.patch

set(_CL_HPP "${CMAKE_CURRENT_SOURCE_DIR}/cl.hpp")
# TODO apple
add_custom_command(OUTPUT "${_CL_HPP}"
    COMMAND patch -i "${CMAKE_SOURCE_DIR}/tools/patches/khronos_cl.hpp.patch"
     -o "${_CL_HPP}"
     "${CMAKE_CURRENT_SOURCE_DIR}/cl.hpp.in" )

add_custom_target("patched_cl_hpp" ALL DEPENDS "${_CL_HPP}" COMMENT "Patching cl.hpp")

if(INSTALL_OPENCL_HEADERS)
  install(FILES cl.h cl_ext.h  cl_gl.h cl_gl_ext.h cl_platform.h opencl.h
          DESTINATION "${POCL_INSTALL_OPENCL_HEADER_DIR}")
endif()
