
if(HAVE_LIBJPEG_TURBO)
    list(APPEND POCL_DEVICES_LINK_LIST PkgConfig::LIBJPEG_TURBO)
endif()

list(APPEND POCL_DEVICES_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/pocl_dbk_khr_jpeg_cpu.c ${CMAKE_CURRENT_SOURCE_DIR}/pocl_dbk_khr_jpeg_cpu.h
)

set(POCL_DEVICES_LINK_LIST ${POCL_DEVICES_LINK_LIST} PARENT_SCOPE)
set(POCL_DEVICES_SOURCES ${POCL_DEVICES_SOURCES} PARENT_SCOPE)


