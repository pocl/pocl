#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2014 pocl developers
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

# cmake version of lib/kernel/rules.mk

separate_arguments(KERNEL_CL_FLAGS)
separate_arguments(KERNEL_CLANGXX_FLAGS)

#/usr/bin/clang --target=x86_64-pc-linux-gnu -march=bdver1 -Xclang -ffake-address-space-map -emit-llvm -ffp-contract=off -D__OPENCL_VERSION__=120 -DPOCL_VECMATHLIB_BUILTIN -D__CBUILD__ -o get_local_id.bc -c ${CMAKE_SOURCE_DIR}/lib/kernel/get_local_id.c -include ${CMAKE_SOURCE_DIR}/include/_kernel_c.h
#	  @CLANG@ ${CLANG_FLAGS} ${KERNEL_CL_FLAGS} -D__CBUILD__ -c -o $@ -include ${abs_top_srcdir}/include/_kernel_c.h $< 
function(compile_c_to_bc FILENAME BC_FILE_LIST)
    set(BC_FILE "${FILENAME}.bc")
    string(REPLACE "vecmathlib-pocl/" "" BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${BC_FILE}")
    set(${BC_FILE_LIST} ${${BC_FILE_LIST}} ${BC_FILE} PARENT_SCOPE)
    set(FULL_F_PATH "${CMAKE_SOURCE_DIR}/lib/kernel/${FILENAME}")

    add_custom_command( OUTPUT "${BC_FILE}"
        DEPENDS "${FULL_F_PATH}"
        "${CMAKE_SOURCE_DIR}/include/pocl_types.h"
        "${CMAKE_SOURCE_DIR}/include/pocl_features.h"
        "${CMAKE_SOURCE_DIR}/include/_kernel_c.h"
        ${KERNEL_DEPEND_HEADERS}
        COMMAND "${CLANG}" ${CLANG_FLAGS} ${KERNEL_CL_FLAGS} "-D__CBUILD__" "-o" "${BC_FILE}" "-c" "${FULL_F_PATH}" "-include" "${CMAKE_SOURCE_DIR}/include/_kernel_c.h"
        COMMENT "Building C to LLVM bitcode ${BC_FILE}" 
        VERBATIM)
endfunction()

# /usr/bin/clang++ --target=x86_64-pc-linux-gnu -march=bdver1 -Xclang -ffake-address-space-map -emit-llvm -ffp-contract=off -DVML_NO_IOSTREAM -DPOCL_VECMATHLIB_BUILTIN -o trunc.bc -c ${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib-pocl/trunc.cc -include ${CMAKE_SOURCE_DIR}/include/pocl_features.h
# 	@CLANGXX@ ${CLANG_FLAGS} ${KERNEL_CLANGXX_FLAGS} -c -o $@ $< -include ${abs_top_srcdir}/include/pocl_features.h
function(compile_cc_to_bc FILENAME BC_FILE_LIST)
    set(BC_FILE "${FILENAME}.bc")
    string(REPLACE "vecmathlib-pocl/" "" BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${BC_FILE}")
    set(${BC_FILE_LIST} ${${BC_FILE_LIST}} ${BC_FILE} PARENT_SCOPE)
    set(FULL_F_PATH "${CMAKE_SOURCE_DIR}/lib/kernel/${FILENAME}")

    #MESSAGE(STATUS "BC_FILE: ${BC_FILE}")

    add_custom_command(OUTPUT "${BC_FILE}"
        DEPENDS "${FULL_F_PATH}"
          "${CMAKE_SOURCE_DIR}/include/pocl_features.h"
          ${KERNEL_DEPEND_HEADERS}
        COMMAND  "${CLANGXX}" ${CLANG_FLAGS} ${KERNEL_CLANGXX_FLAGS} "-o" "${BC_FILE}" "-c" "${FULL_F_PATH}" "-include" "${CMAKE_SOURCE_DIR}/include/pocl_features.h"
        COMMENT "Building C++ to LLVM bitcode ${BC_FILE}" 
        VERBATIM)
endfunction()

# /usr/bin/clang --target=x86_64-pc-linux-gnu -march=bdver1 -Xclang -ffake-address-space-map -emit-llvm -ffp-contract=off -x cl -D__OPENCL_VERSION__=120 -DPOCL_VECMATHLIB_BUILTIN -fsigned-char -o atan2pi.bc -c ${CMAKE_SOURCE_DIR}/lib/kernel/vecmathlib-pocl/atan2pi.cl -include ${CMAKE_SOURCE_DIR}/include/_kernel.h
function(compile_cl_to_bc FILENAME BC_FILE_LIST)
    set(BC_FILE "${FILENAME}.bc")
    string(REPLACE "vecmathlib-pocl/" "" BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${BC_FILE}")
    set(${BC_FILE_LIST} ${${BC_FILE_LIST}} ${BC_FILE} PARENT_SCOPE)
    set(FULL_F_PATH "${CMAKE_SOURCE_DIR}/lib/kernel/${FILENAME}")

    #MESSAGE(STATUS "BC_FILE: ${BC_FILE}")

    add_custom_command( OUTPUT "${BC_FILE}"
        DEPENDS "${FULL_F_PATH}"
          "${CMAKE_SOURCE_DIR}/include/_kernel.h"
          "${CMAKE_SOURCE_DIR}/include/_kernel_c.h"
          "${CMAKE_SOURCE_DIR}/include/pocl_types.h" 
          "${CMAKE_SOURCE_DIR}/include/pocl_features.h"
          ${KERNEL_DEPEND_HEADERS}
        COMMAND "${CLANG}" ${CLANG_FLAGS} "-x" "cl" ${KERNEL_CL_FLAGS}  "-fsigned-char"  "-o" "${BC_FILE}" "-c" "${FULL_F_PATH}" "-include" "${CMAKE_SOURCE_DIR}/include/_kernel.h"
        COMMENT "Building CL to LLVM bitcode ${BC_FILE}" 
        VERBATIM)
endfunction()


function(compile_ll_to_bc FILENAME BC_FILE_LIST)
    set(BC_FILE "${FILENAME}.bc")
    string(REPLACE "vecmathlib-pocl/" "" BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${BC_FILE}")
    set(${BC_FILE_LIST} ${${BC_FILE_LIST}} ${BC_FILE} PARENT_SCOPE)
    set(FULL_F_PATH "${CMAKE_SOURCE_DIR}/lib/kernel/${FILENAME}")


    add_custom_command( OUTPUT "${BC_FILE}"
        DEPENDS ""
        COMMAND "${LLVM_AS}" "-o" "${BC_FILE}" "${CMAKE_CURRENT_SOURCE_DIR}/../${FILENAME}"
        COMMENT "Building LL to LLVM bitcode ${BC_FILE}" 
        VERBATIM)
endfunction()


macro(compile_to_bc OUTPUT_FILE_LIST)
  foreach(FILENAME ${ARGN})
  if(FILENAME MATCHES "[.]c$")
    compile_c_to_bc("${FILENAME}" ${OUTPUT_FILE_LIST})
  elseif(FILENAME MATCHES "[.]cc$")
    compile_cc_to_bc("${FILENAME}" ${OUTPUT_FILE_LIST})
  elseif(FILENAME MATCHES "[.]cl$")
    compile_cl_to_bc("${FILENAME}" ${OUTPUT_FILE_LIST})
  elseif(FILENAME MATCHES "[.]ll$")
    compile_ll_to_bc("${FILENAME}" ${OUTPUT_FILE_LIST})
  else()
    message(FATAL_ERROR "Dont know how to compile ${FILENAME} to .bc !")
  endif()
  endforeach()
endmacro()



function(make_kernel_bc OUTPUT_VAR NAME)
  set(KERNEL_BC "kernel-${NAME}.bc")
  set(${OUTPUT_VAR} "${KERNEL_BC}" PARENT_SCOPE)

  compile_to_bc(BC_LIST ${ARGN})

  # fix too long commandline with cat and xargs
  SET(BC_LIST_FILE_TXT "")
  foreach(FILENAME ${BC_LIST})
    # straight parsing semicolon separated list with xargs -d didn't work on windows.. no such switch available
    SET(BC_LIST_FILE_TXT "${BC_LIST_FILE_TXT} \"${FILENAME}\"")
  endforeach()
  SET (BC_LIST_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/kernel_linklist.txt")
  FILE (WRITE "${BC_LIST_FILE}" "${BC_LIST_FILE_TXT}")

  add_custom_command( OUTPUT "${KERNEL_BC}"
# ${KERNEL_BC}: ${OBJ}
        DEPENDS ${BC_LIST}
#	    @LLVM_LINK@ $^ -o - | @LLVM_OPT@ ${LLC_FLAGS} ${KERNEL_LIB_OPT_FLAGS} -O3 -fp-contract=off -o $@
        COMMAND "${XARGS_EXEC}" "${LLVM_LINK}" "-o" "kernel-${NAME}-unoptimized.bc" < "${BC_LIST_FILE}"
        COMMAND "${LLVM_OPT}" ${LLC_FLAGS} "-O3" "-fp-contract=off" "-o" "${KERNEL_BC}" "kernel-${NAME}-unoptimized.bc"
        COMMENT "Linking Kernel bitcode ${KERNEL_BC}" 
        VERBATIM)

endfunction()

