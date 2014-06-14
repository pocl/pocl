#~ add_library("pocl-devices-tce-ttasim" OBJECT ttasim.h ttasim.cc)
#~ set_target_properties("pocl-devices-tce-ttasim" PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(POCL_DEVICE_SOURCES ${POCL_DEVICE_SOURCES}
  devices/tce/ttasim/ttasim.h
  devices/tce/ttasim/ttasim.cc PARENT_SCOPE)

set_source_files_properties(devices/tce/ttasim/ttasim.h
  devices/tce/ttasim/ttasim.cc PROPERTIES COMPILE_FLAGS "${TCE_INCLUDES}")

# TODO flags...
