if(ENABLE_TCEMC)
  add_subdirectory("ttasim")
endif()

set(POCL_DEVICE_SOURCES ${POCL_DEVICE_SOURCES}
  devices/tce/tce_common.h
  devices/tce/tce_common.cc PARENT_SCOPE)

set_source_files_properties(devices/tce/tce_common.h
  devices/tce/tce_common.cc PROPERTIES COMPILE_FLAGS "${TCE_INCLUDES}")


#
# dist_pkgdata_DATA = tta_device_main.c
install(FILES "tta_device_main.c" "tta_device_main_dthread.c"
        DESTINATION ${POCL_INSTALL_PRIVATE_HEADER_DIR})
