
add_subdirectory("pthread")
add_subdirectory("basic")


add_subdirectory("topology")
set(POCL_PUBLIC_LINK_LIST ${HWLOC_LIBRARIES})

if(ENABLE_TCE)
  add_subdirectory("tce")
  list(APPEND POCL_PUBLIC_LINK_LIST ${TCE_LIBS})
endif()

if(ENABLE_SPU)
  add_subdirectory("cellspu")
  list(APPEND POCL_PUBLIC_LINK_LIST "spe2")
endif()


set(POCL_PUBLIC_LINK_LIST ${POCL_PUBLIC_LINK_LIST} PARENT_SCOPE)
set(POCL_DEVICE_SOURCES ${POCL_DEVICE_SOURCES}
        devices/devices.h  devices/devices.c
        devices/bufalloc.c  devices/dev_image.h
        devices/common.h devices/common.c
        devices/bufalloc.h  devices/cpuinfo.c devices/cpuinfo.h
        PARENT_SCOPE)


