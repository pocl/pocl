add_library("pocl-devices-cellspu" OBJECT cellspu.h cellspu.c)
set(POCL_DEVICES_OBJS "${POCL_DEVICES_OBJS};$<TARGET_OBJECTS:pocl-devices-cellspu>" PARENT_SCOPE)
