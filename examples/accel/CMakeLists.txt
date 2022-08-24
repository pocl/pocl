#=============================================================================
#   CMake build system files
#
#   Copyright (c) 2019 Pekka Jääskeläinen
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

set(PROGRAMS_TO_BUILD accel_example)

if (MSVC)
  add_compile_options(${OPENCL_CFLAGS})
else ()
  add_compile_options("-std=c++11" "-Wno-deprecated" "-Wno-deprecated-declarations" ${OPENCL_CFLAGS})
endif ()

add_definitions("-DSRCDIR=\"${CMAKE_CURRENT_SOURCE_DIR}\"")
include_directories("${CMAKE_SOURCE_DIR}/lib/CL" ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 14)
foreach(PROG ${PROGRAMS_TO_BUILD})
  add_executable("${PROG}" "${PROG}.cpp")
  target_link_libraries("${PROG}" ${POCLU_LINK_OPTIONS})
endforeach()

# this test requires 2 devices
add_executable(accel_countred accel_countred.cpp OpenCLcontext.cpp  OpenCLcontext.h)
target_link_libraries(accel_countred ${POCLU_LINK_OPTIONS})


add_custom_target(bitstreams)

if(ENABLE_TCE)
  #add_subdirectory(gen_accel)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/firmwares)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/bitstreams)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/firmware_imgs)



  configure_file(${CMAKE_SOURCE_DIR}/tools/data/tta_test_machines/axim_sep.adf ${CMAKE_CURRENT_BINARY_DIR}/firmwares)
  configure_file(${CMAKE_SOURCE_DIR}/tools/data/tta_test_machines/axim_fused_global_private.adf ${CMAKE_CURRENT_BINARY_DIR}/firmwares)
  configure_file(${CMAKE_SOURCE_DIR}/tools/data/tta_test_machines/axim_fused_global_cq.adf ${CMAKE_CURRENT_BINARY_DIR}/firmwares)
  configure_file(${CMAKE_SOURCE_DIR}/tools/data/tta_test_machines/axim_fused_global_cq_private.adf ${CMAKE_CURRENT_BINARY_DIR}/firmwares)

  configure_file(${CMAKE_SOURCE_DIR}/tools/data/tta_test_machines/relative_sep.adf ${CMAKE_CURRENT_BINARY_DIR}/firmwares)
  configure_file(${CMAKE_SOURCE_DIR}/tools/data/tta_test_machines/relative_fused_global_private.adf ${CMAKE_CURRENT_BINARY_DIR}/firmwares)
  configure_file(${CMAKE_SOURCE_DIR}/tools/data/tta_test_machines/relative_fused_global_cq.adf ${CMAKE_CURRENT_BINARY_DIR}/firmwares)
  configure_file(${CMAKE_SOURCE_DIR}/tools/data/tta_test_machines/relative_fused_global_cq_private.adf ${CMAKE_CURRENT_BINARY_DIR}/firmwares)

  configure_file(firmware.c ${CMAKE_CURRENT_BINARY_DIR})

  function(generate_firmware core_name base_address queue_length queue_start init_sp data_start use_axim)
    set(ADF ${CMAKE_SOURCE_DIR}/tools/data/tta_test_machines/${core_name}.adf)
    set(TPEF ${CMAKE_CURRENT_BINARY_DIR}/firmwares/${core_name}.tpef)

    message("ADF=${ADF}")

    set(PARAMS "")
    if(NOT base_address STREQUAL "")
      list(APPEND PARAMS "-DBASE_ADDRESS=${base_address}")
    endif()
    if(NOT queue_length STREQUAL "")
      list(APPEND PARAMS "-DQUEUE_LENGTH=${queue_length}")
    endif()
    if(NOT queue_start STREQUAL "")
      list(APPEND PARAMS "-DQUEUE_START=${queue_start}")
    endif()
    if(NOT init_sp STREQUAL "")
      list(APPEND PARAMS "--init-sp=${init_sp}")
      list(APPEND PARAMS "--data-start=${data_start}")
    endif()
    add_custom_command(OUTPUT firmwares/${core_name}.tpef
      COMMAND tcecc -o3 ${PARAMS} -g -a ${ADF} -o ${TPEF} firmware.c
      COMMAND generatebits -e tta_core -f bin2n -p ${TPEF} ${ADF}
      COMMAND tcedisasm -n ${ADF} ${TPEF}
      COMMAND mv ${core_name}.img ${CMAKE_CURRENT_BINARY_DIR}/firmware_imgs/
      DEPENDS ${ADF} ${CMAKE_CURRENT_SOURCE_DIR}/firmware.c
      COMMENT "Compiling ${core_name} firmware with params ${PARAMS}")
    add_custom_target(${core_name}_firmware ALL DEPENDS firmwares/${core_name}.tpef)
    
    
    add_custom_command(OUTPUT rtls/rtl_${core_name}
      COMMAND rm -rf rtls/rtl_${core_name}
      COMMAND generateprocessor --prefer-generation -o rtls/rtl_${core_name} -e tta_core --icd-arg-list=debugger:minimal,fpga-optimized:yes
      --hdb-list=generate_lsu_32.hdb,generate_rf_iu.hdb,xilinx_series7.hdb,generate_base32.hdb
      -f onchip -d onchip -g AlmaIFIntegrator -p ${TPEF} ${ADF}
      COMMAND generatebits -x rtls/rtl_${core_name} -e tta_core ${ADF}
      DEPENDS ${ADF})
    add_custom_target(${core_name}_rtl ALL DEPENDS rtls/rtl_${core_name})


    add_custom_command(OUTPUT bitstreams/${core_name}.bit
      COMMAND rm -rf vivado_${core_name}_1
      COMMAND ${VIVADO_PATH}vivado -mode batch -source ${CMAKE_CURRENT_SOURCE_DIR}/generate_project.tcl
       -tclargs ${core_name} -tclargs 1 -tclargs ${use_axim}
       COMMAND cp vivado_${core_name}_1/vivado_${core_name}_1.runs/impl_1/toplevel_wrapper.bit
        bitstreams/${core_name}.bit
      COMMAND cp vivado_${core_name}_1/vivado_${core_name}_1.gen/sources_1/bd/toplevel/hw_handoff/toplevel.hwh
        bitstreams/${core_name}.hwh
      DEPENDS ${ADF} ${CMAKE_CURRENT_SOURCE_DIR}/generate_project.tcl)
    add_custom_target(${core_name}_bs DEPENDS bitstreams/${core_name}.bit)

    add_dependencies(bitstreams ${core_name}_bs)

  endfunction()

  
  generate_firmware("axim_sep"                         "0x43C00000" "1023" ""           ""           ""           "1")
  generate_firmware("axim_fused_global_cq"             "0x43C00000" "3"    "0x43C1FF00" ""           ""           "1")
  generate_firmware("axim_fused_global_private"        "0x43C00000" "1023" ""           "0x43C20000" "0x43C1F800" "1")
  generate_firmware("axim_fused_global_cq_private"     "0x43C00000" "3"    "0x43C1F700" "0x43C20000" "0x43C1F800" "1")
  generate_firmware("relative_sep"                     ""           "1023" ""           ""           ""           "0")
  generate_firmware("relative_fused_global_cq"         ""           "3"    "0x7F00"     ""           ""           "0")
  generate_firmware("relative_fused_global_private"    ""           "1023" ""           "0x8000"     "0x7800"     "0")
  generate_firmware("relative_fused_global_cq_private" ""           "3"    "0x7700"     "0x8000"     "0x7800"     "0")



endif()

if(DEFINED VITIS_HLS_PATH)
  add_subdirectory(hls)
endif()

######################################################################

add_test_pocl(NAME "examples/accel/countred" COMMAND "accel_example" "pocl.countred")

add_test_pocl(NAME "examples/accel/add.i32" COMMAND "accel_example" "pocl.add.i32")

add_test_pocl(NAME "examples/accel/mul.i32" COMMAND "accel_example" "pocl.mul.i32")

add_test_pocl(NAME "examples/accel/abs.f32" COMMAND "accel_example" "pocl.abs.f32")

set_tests_properties( "examples/accel/countred" "examples/accel/add.i32"
  "examples/accel/mul.i32"  "examples/accel/abs.f32"
  PROPERTIES
    PASS_REGULAR_EXPRESSION "OK"
    LABELS "custom_device;accel_no_LLVM"
    DEPENDS "pocl_version_check")
