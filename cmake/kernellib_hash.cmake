# TODO this is duplicated in top CMakeLists.txt
function(rename_if_different SRC DST)
  if(EXISTS "${DST}")
    file(MD5 "${SRC}" OLD_MD5)
    file(MD5 "${DST}" NEW_MD5)
    if(NOT OLD_MD5 STREQUAL NEW_MD5)
      message(STATUS "Renaming ${SRC} to ${DST}")
      file(RENAME "${SRC}" "${DST}")
    endif()
  else()
    message(STATUS "Renaming ${SRC} to ${DST}")
    file(RENAME "${SRC}" "${DST}")
  endif()
endfunction()


string(REPLACE "****" ";" KERNEL_BC_LIST "${KERNEL_BC_LIST_ESCAPED}")
foreach(KERNEL_BC IN LISTS KERNEL_BC_LIST)
  if(EXISTS ${KERNEL_BC})
    file(SHA1 "${KERNEL_BC}" S)
    set(S1 "${S}__${S1}")
  endif()
endforeach()

file(SHA1 "${INCLUDEDIR}/_kernel.h" S2)
file(SHA1 "${INCLUDEDIR}/_kernel_c.h" S3)
file(SHA1 "${INCLUDEDIR}/pocl_types.h" S4)

file(WRITE "${OUTPUT}.new" "#define POCL_KERNELLIB_SHA1 \"${S1}${S2}_${S3}_${S4}\"")

rename_if_different("${OUTPUT}.new" "${OUTPUT}")
