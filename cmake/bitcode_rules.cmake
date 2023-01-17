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

separate_arguments(KERNEL_C_FLAGS)
separate_arguments(KERNEL_CL_FLAGS)
separate_arguments(KERNEL_CXX_FLAGS)

function(compile_c_to_bc FILENAME SUBDIR BC_FILE_LIST)
    get_filename_component(FNAME "${FILENAME}" NAME)
    set(BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SUBDIR}/${FNAME}.bc")
    set(${BC_FILE_LIST} ${${BC_FILE_LIST}} ${BC_FILE} PARENT_SCOPE)
    if(IS_ABSOLUTE "${FILENAME}")
      set(FULL_F_PATH "${FILENAME}")
    else()
      set(FULL_F_PATH "${CMAKE_SOURCE_DIR}/lib/kernel/${FILENAME}")
    endif()

    add_custom_command( OUTPUT "${BC_FILE}"
        DEPENDS "${FULL_F_PATH}"
        "${CMAKE_SOURCE_DIR}/include/pocl_types.h"
        "${CMAKE_SOURCE_DIR}/include/_kernel_c.h"
        COMMAND "${CLANG}" ${CLANG_FLAGS} ${DEVICE_CL_FLAGS} "-O1"
        ${KERNEL_C_FLAGS} "-o" "${BC_FILE}" "-c" "${FULL_F_PATH}"
        "-I${CMAKE_SOURCE_DIR}/include"
        "-include" "${CMAKE_SOURCE_DIR}/include/_kernel_c.h"
        COMMENT "Building C to LLVM bitcode ${BC_FILE}"
        VERBATIM)
endfunction()

function(compile_cc_to_bc FILENAME SUBDIR BC_FILE_LIST)
    get_filename_component(FNAME "${FILENAME}" NAME)
    set(BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SUBDIR}/${FNAME}.bc")
    set(${BC_FILE_LIST} ${${BC_FILE_LIST}} ${BC_FILE} PARENT_SCOPE)
    if(IS_ABSOLUTE "${FILENAME}")
      set(FULL_F_PATH "${FILENAME}")
    else()
      set(FULL_F_PATH "${CMAKE_SOURCE_DIR}/lib/kernel/${FILENAME}")
    endif()

    add_custom_command(OUTPUT "${BC_FILE}"
        DEPENDS "${FULL_F_PATH}"
        COMMAND  "${CLANGXX}" ${CLANG_FLAGS} ${KERNEL_CXX_FLAGS}
        ${DEVICE_C_FLAGS} "-o" "${BC_FILE}" "-c" "${FULL_F_PATH}" "-O1"
        COMMENT "Building C++ to LLVM bitcode ${BC_FILE}"
        VERBATIM)
endfunction()

function(compile_cl_to_bc FILENAME SUBDIR BC_FILE_LIST EXTRA_CONFIG)
    get_filename_component(FNAME "${FILENAME}" NAME)
    get_filename_component(FNAME_WE "${FILENAME}" NAME_WE)
    set(BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SUBDIR}/${FNAME}.bc")
    set(${BC_FILE_LIST} ${${BC_FILE_LIST}} ${BC_FILE} PARENT_SCOPE)
    if(IS_ABSOLUTE "${FILENAME}")
      set(FULL_F_PATH "${FILENAME}")
    else()
      set(FULL_F_PATH "${CMAKE_SOURCE_DIR}/lib/kernel/${FILENAME}")
    endif()

    set(DEPENDLIST
          "${CMAKE_SOURCE_DIR}/include/_kernel.h"
          "${CMAKE_SOURCE_DIR}/include/_kernel_c.h"
          "${CMAKE_SOURCE_DIR}/include/pocl_types.h")
    set(INCLUDELIST
        "-include" "${CMAKE_SOURCE_DIR}/include/_kernel.h"
        "-include" "${CMAKE_SOURCE_DIR}/include/_enable_all_exts.h")

    if(FILENAME MATCHES "sleef")
      list(APPEND DEPENDLIST
          "${EXTRA_CONFIG}"
          )
      list(APPEND DEPENDLIST ${SLEEF_CL_KERNEL_DEPEND_HEADERS})
      list(APPEND INCLUDELIST
        "-DMAX_PRECISION"
        "-I" "${CMAKE_SOURCE_DIR}/lib/kernel/sleef/include" # for sleef_cl.h
        "-include" "${EXTRA_CONFIG}")
    endif()

    if(FILENAME MATCHES "libclc")
      list(APPEND DEPENDLIST ${LIBCLC_KERNEL_DEPEND_HEADERS})

      set(I32 "${CMAKE_SOURCE_DIR}/lib/kernel/libclc/${FNAME_WE}_fp32.cl")
      if(EXISTS "${I32}")
        list(APPEND DEPENDLIST "${I32}")
      endif()

      set(I64 "${CMAKE_SOURCE_DIR}/lib/kernel/libclc/${FNAME_WE}_fp64.cl")
      if(EXISTS "${I64}")
        list(APPEND DEPENDLIST "${I64}")
      endif()

      list(APPEND INCLUDELIST
        "-I" "${CMAKE_SOURCE_DIR}/lib/kernel/libclc")
    endif()

    add_custom_command( OUTPUT "${BC_FILE}"
        DEPENDS "${FULL_F_PATH}"
          ${DEPENDLIST}
        COMMAND "${CLANG}" ${CLANG_FLAGS}
        ${KERNEL_CL_FLAGS} ${DEVICE_CL_FLAGS}
        "-o" "${BC_FILE}" "-c" "${FULL_F_PATH}"
        ${INCLUDELIST}
        COMMENT "Building CL to LLVM bitcode ${BC_FILE}"
        VERBATIM)
endfunction()

# ARGN - extra defines / arguments to clang
# can't use c_to_bc, since SLEEF's C files need to be prefixed with EXT
# (because the same files are compiled multiple times)
function(compile_sleef_c_to_bc EXT FILENAME SUBDIR BCLIST)
    get_filename_component(FNAME "${FILENAME}" NAME)
    set(BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SUBDIR}/${EXT}_${FNAME}.bc")
    list(APPEND ${BCLIST} "${BC_FILE}")
    set(${BCLIST} ${${BCLIST}} PARENT_SCOPE)
    if(IS_ABSOLUTE "${FILENAME}")
      set(FULL_F_PATH "${FILENAME}")
    else()
      set(FULL_F_PATH "${CMAKE_SOURCE_DIR}/lib/kernel/${FILENAME}")
    endif()

    add_custom_command( OUTPUT "${BC_FILE}"
        DEPENDS "${FULL_F_PATH}"
        ${SLEEF_C_KERNEL_DEPEND_HEADERS}
        COMMAND "${CLANG}" ${CLANG_FLAGS} ${DEVICE_C_FLAGS} ${KERNEL_C_FLAGS} ${ARGN}
        "-I" "${CMAKE_SOURCE_DIR}/lib/kernel/sleef/arch"
        "-I" "${CMAKE_SOURCE_DIR}/lib/kernel/sleef/libm"
        "-I" "${CMAKE_SOURCE_DIR}/lib/kernel/sleef/include"
        "-O1" "-o" "${BC_FILE}" "-c" "${FULL_F_PATH}"
        COMMENT "Building SLEEF to LLVM bitcode ${BC_FILE}"
        VERBATIM)
endfunction()

# compiles LLVM IR in text-format (an .ll file) to LLVM IR bitcode (binary format, .bc)
# BCLIST is the name of a list variable; the path of the generated BC file will be
# appended to this variable is the caller's scope
function(compile_ll_to_bc FILENAME SUBDIR BCLIST)
    if(IS_ABSOLUTE "${FILENAME}")
      set(FULL_F_PATH "${FILENAME}")
    else()
      set(FULL_F_PATH "${CMAKE_SOURCE_DIR}/lib/kernel/${FILENAME}")
    endif()
    get_filename_component(FNAME "${FILENAME}" NAME)
    set(BC_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SUBDIR}/${FNAME}.bc")
    list(APPEND ${BCLIST} "${BC_FILE}")
    set(${BCLIST} ${${BCLIST}} PARENT_SCOPE)

    if(LLVM_VERSION VERSION_EQUAL 15.0)
      # both of these are necesssary. some of the files (like barrier.ll)
      # don't contain any pointers and thus cannot be guessed; if llvm-as
      # produces the wrong opaque-type file, later llvm-link will fail
      if(ENABLE_LLVM_OPAQUE_POINTERS)
        set(EXTRA_OPT "-opaque-pointers=1")
      else()
        set(EXTRA_OPT "-opaque-pointers=0")
      endif()
    endif()

    add_custom_command( OUTPUT "${BC_FILE}"
        DEPENDS "${FULL_F_PATH}"
        COMMAND "${LLVM_AS}" ${EXTRA_OPT} "-o" "${BC_FILE}" "${FULL_F_PATH}"
        COMMENT "Building LL to LLVM bitcode ${BC_FILE}" 
        VERBATIM)
endfunction()


macro(compile_to_bc SUBDIR OUTPUT_FILE_LIST EXTRA_CONFIG)
  foreach(FILENAME ${ARGN})
  if(FILENAME MATCHES "[.]c$")
    compile_c_to_bc("${FILENAME}" "${SUBDIR}" ${OUTPUT_FILE_LIST})
  elseif(FILENAME MATCHES "[.]cc$")
    compile_cc_to_bc("${FILENAME}" "${SUBDIR}" ${OUTPUT_FILE_LIST})
  elseif(FILENAME MATCHES "[.]cl$")
    compile_cl_to_bc("${FILENAME}" "${SUBDIR}" ${OUTPUT_FILE_LIST} "${EXTRA_CONFIG}")
  elseif(FILENAME MATCHES "[.]ll$")
    compile_ll_to_bc("${FILENAME}" "${SUBDIR}" ${OUTPUT_FILE_LIST})
  else()
    message(FATAL_ERROR "Dont know how to compile ${FILENAME} to .bc !")
  endif()
  endforeach()
endmacro()

function(generate_cuda_spir_wrapper OUTPUT)
  set(FNAME "${CMAKE_CURRENT_BINARY_DIR}/spir_wrapper.ll")
  set(${OUTPUT} "${FNAME}" PARENT_SCOPE)
  if(ENABLE_LLVM_OPAQUE_POINTERS)
    set(EXTRA_OPT "--opaque-pointers")
  else()
    unset(EXTRA_OPT)
  endif()

  add_custom_command( OUTPUT "${FNAME}"
      DEPENDS "${CMAKE_SOURCE_DIR}/lib/kernel/SPIR/generate_spir_wrapper.py"
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/lib/kernel/SPIR/generate_spir_wrapper.py" ${EXTRA_OPT} "-t" "cuda" "${FNAME}"
      COMMENT "Generating CUDA SPIR wrapper to ${FNAME}"
      VERBATIM)
endfunction()

function(generate_cpu_spir_wrapper ARCH SUBDIR SIZE OUTPUT)
  set(FNAME "${CMAKE_CURRENT_BINARY_DIR}/${SUBDIR}/spir_wrapper_${SIZE}bit.ll")
  set(${OUTPUT} "${FNAME}" PARENT_SCOPE)
  if(ENABLE_LLVM_OPAQUE_POINTERS)
    set(EXTRA_OPT "--opaque-pointers")
  else()
    unset(EXTRA_OPT)
  endif()

  add_custom_command( OUTPUT "${FNAME}"
      DEPENDS "${CMAKE_SOURCE_DIR}/lib/kernel/SPIR/generate_spir_wrapper.py"
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/lib/kernel/SPIR/generate_spir_wrapper.py" "-g" ${EXTRA_OPT} "-t" "${ARCH}" "-r" "${SIZE}" "${FNAME}"
      COMMENT "Generating x86-64 ${VECSIZE}-bit wrapper for ${SUBDIR} to ${FNAME}"
      VERBATIM)
endfunction()

function(generate_opaque_ptr_ll FILENAME SUBDIR OUTPUT)
  get_filename_component(FNAME "${FILENAME}" NAME)
  set(LL_FILE "${CMAKE_CURRENT_BINARY_DIR}/${SUBDIR}/${FNAME}")
  set(${OUTPUT} "${LL_FILE}" PARENT_SCOPE)
  if(IS_ABSOLUTE "${FILENAME}")
    set(FULL_F_PATH "${FILENAME}")
  else()
    set(FULL_F_PATH "${CMAKE_SOURCE_DIR}/lib/kernel/${FILENAME}")
  endif()

  add_custom_command( OUTPUT "${LL_FILE}"
      DEPENDS "${FULL_F_PATH}"
      COMMAND "${CMAKE_COMMAND}"
      "-DINPUT_FILE=${FULL_F_PATH}" "-DOUTPUT_FILE=${LL_FILE}"
      -P "${CMAKE_SOURCE_DIR}/cmake/make_opaque_ptr.cmake"
      COMMENT "Generating opaque-pointer version of ${FNAME}"
      VERBATIM)

endfunction()


function(make_kernel_bc OUTPUT_VAR NAME SUBDIR USE_SLEEF EXTRA_BC EXTRA_CONFIG)
  set(KERNEL_BC "${CMAKE_CURRENT_BINARY_DIR}/kernel-${NAME}.bc")
  set(${OUTPUT_VAR} "${KERNEL_BC}" PARENT_SCOPE)

  file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/${SUBDIR}")
  compile_to_bc("${SUBDIR}" BC_LIST "${EXTRA_CONFIG}" ${ARGN})

  set(DEPENDLIST ${BC_LIST})
  # fix too long commandline with cat and xargs
  set(BC_LIST_FILE_TXT "")
  foreach(FILENAME ${BC_LIST})
    # straight parsing semicolon separated list with xargs -d didn't work on windows.. no such switch available
    set(BC_LIST_FILE_TXT "${BC_LIST_FILE_TXT} \"${FILENAME}\"")
  endforeach()
  if(USE_SLEEF)
    set(BC_LIST_FILE_TXT "${BC_LIST_FILE_TXT} \"${EXTRA_BC}\"")
    list(APPEND DEPENDLIST ${EXTRA_BC} "sleef_config_${VARIANT}")
  endif()
  set(BC_LIST_FILE "${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/kernel_${NAME}_linklist.txt")
  file(WRITE "${BC_LIST_FILE}" "${BC_LIST_FILE_TXT}")

  # don't waste time optimizing the kernels IR when in developer mode
  if(DEVELOPER_MODE)
    set(LINK_OPT_COMMAND COMMAND "${XARGS_EXEC}" "${LLVM_LINK}" "-o" "${KERNEL_BC}" < "${BC_LIST_FILE}")
  else()
    set(LINK_CMD COMMAND "${XARGS_EXEC}" "${LLVM_LINK}" "-o" "kernel-${NAME}-unoptimized.bc" < "${BC_LIST_FILE}")
    set(OPT_CMD COMMAND "${LLVM_OPT}" ${LLC_FLAGS} "-O3" "-fp-contract=off" "-o" "${KERNEL_BC}" "kernel-${NAME}-unoptimized.bc")
    set(LINK_OPT_COMMAND ${LINK_CMD} ${OPT_CMD})
  endif()

  add_custom_command( OUTPUT "${KERNEL_BC}"
        DEPENDS ${DEPENDLIST}
        ${LINK_OPT_COMMAND}
        COMMENT "Linking & optimizing Kernel bitcode ${KERNEL_BC}"
        VERBATIM)

endfunction()

