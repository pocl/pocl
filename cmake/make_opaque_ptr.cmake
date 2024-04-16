# takes a LLVM IR file in text (.LL) format and produces a copy with pointers
# turned into opaque pointers. Only good enough to be used on PoCL's
# builtin library.
#
# some argument checking:
# test_cmd is the command to run with all its arguments, separated by "####"
if( NOT INPUT_FILE )
  message( FATAL_ERROR "Variable INPUT_FILE not defined" )
endif()

if( NOT OUTPUT_FILE )
  message( FATAL_ERROR "Variable OUTPUT_FILE not defined" )
endif()

file(READ "${INPUT_FILE}" CONTENT)

  string(REPLACE "i8 addrspace(4)*" "ptr addrspace(4)" CONTENT "${CONTENT}")
  string(REPLACE "i16 addrspace(4)*" "ptr addrspace(4)" CONTENT "${CONTENT}")
  string(REPLACE "i32 addrspace(4)*" "ptr addrspace(4)" CONTENT "${CONTENT}")
  string(REPLACE "i64 addrspace(4)*" "ptr addrspace(4)" CONTENT "${CONTENT}")
  string(REPLACE "float addrspace(4)*" "ptr addrspace(4)" CONTENT "${CONTENT}")
  string(REPLACE "double addrspace(4)*" "ptr addrspace(4)" CONTENT "${CONTENT}")

  string(REPLACE "i8 addrspace(3)*" "ptr addrspace(3)" CONTENT "${CONTENT}")
  string(REPLACE "i16 addrspace(3)*" "ptr addrspace(3)" CONTENT "${CONTENT}")
  string(REPLACE "i32 addrspace(3)*" "ptr addrspace(3)" CONTENT "${CONTENT}")
  string(REPLACE "i64 addrspace(3)*" "ptr addrspace(3)" CONTENT "${CONTENT}")
  string(REPLACE "float addrspace(3)*" "ptr addrspace(3)" CONTENT "${CONTENT}")
  string(REPLACE "double addrspace(3)*" "ptr addrspace(3)" CONTENT "${CONTENT}")

  string(REPLACE "i8 addrspace(2)*" "ptr addrspace(2)" CONTENT "${CONTENT}")
  string(REPLACE "i16 addrspace(2)*" "ptr addrspace(2)" CONTENT "${CONTENT}")
  string(REPLACE "i32 addrspace(2)*" "ptr addrspace(2)" CONTENT "${CONTENT}")
  string(REPLACE "i64 addrspace(2)*" "ptr addrspace(2)" CONTENT "${CONTENT}")
  string(REPLACE "float addrspace(2)*" "ptr addrspace(2)" CONTENT "${CONTENT}")
  string(REPLACE "double addrspace(2)*" "ptr addrspace(2)" CONTENT "${CONTENT}")

  string(REPLACE "i8 addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "i16 addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "i32 addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "i64 addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "float addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "double addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")

  string(REPLACE "%opencl.image1d_buffer_rw_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image1d_buffer_ro_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image1d_buffer_wo_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image2d_array_rw_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image2d_array_ro_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image2d_array_wo_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image1d_array_rw_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image1d_array_ro_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image1d_array_wo_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")

  string(REPLACE "%opencl.image3d_rw_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image3d_ro_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image3d_wo_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")

  string(REPLACE "%opencl.image2d_rw_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image2d_ro_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image2d_wo_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")

  string(REPLACE "%opencl.image1d_rw_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image1d_ro_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image1d_wo_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")

  string(REPLACE "%opencl.sampler_t addrspace(1)*" "ptr addrspace(1)" CONTENT "${CONTENT}")
  string(REPLACE "%opencl.event_t* addrspace(4)*" "ptr addrspace(4)" CONTENT "${CONTENT}")

  string(REPLACE "i8*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "i16*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "i32*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "i64*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "float*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "double*" "ptr " CONTENT "${CONTENT}")

  string(REPLACE "%opencl.event_t*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "%opencl.sampler_t*" "ptr " CONTENT "${CONTENT}")

  string(REPLACE "%opencl.image1d_wo_t*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image1d_ro_t*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image1d_rw_t*" "ptr " CONTENT "${CONTENT}")

  string(REPLACE "%opencl.image2d_wo_t*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image2d_ro_t*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image2d_rw_t*" "ptr " CONTENT "${CONTENT}")

  string(REPLACE "%opencl.image3d_wo_t*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image3d_ro_t*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image3d_rw_t*" "ptr " CONTENT "${CONTENT}")

  string(REPLACE "%opencl.image1d_buffer_wo_t*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image1d_buffer_ro_t*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image1d_buffer_rw_t*" "ptr " CONTENT "${CONTENT}")

  string(REPLACE "%opencl.image1d_array_wo_t*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image1d_array_ro_t*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image1d_array_rw_t*" "ptr " CONTENT "${CONTENT}")

  string(REPLACE "%opencl.image2d_array_wo_t*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image2d_array_ro_t*" "ptr " CONTENT "${CONTENT}")
  string(REPLACE "%opencl.image2d_array_rw_t*" "ptr " CONTENT "${CONTENT}")

file(WRITE "${OUTPUT_FILE}" "${CONTENT}")
