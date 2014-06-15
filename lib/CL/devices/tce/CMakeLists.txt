add_compile_options(${TCE_INCLUDES})

if(ENABLE_TCEMC)
  add_subdirectory("ttasim")
  set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS};$<TARGET_OBJECTS:pocl-devices-tce-ttasim>" PARENT_SCOPE)
endif()

add_library("pocl-devices-tce" OBJECT tce_common.h tce_common.cc)
set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS};$<TARGET_OBJECTS:pocl-devices-tce>" PARENT_SCOPE)

# dist_pkgdata_DATA = tta_device_main.c
install(FILES "tta_device_main.c" "tta_device_main_dthread.c"
        DESTINATION ${POCL_INSTALL_PRIVATE_HEADER_DIR})
