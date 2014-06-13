
add_subdirectory("CL")

set(PRIVATE_HEADERS  _kernel.h _kernel_c.h pocl_types.h pocl_features.h pocl_device.h pocl.h)

install(FILES ${PRIVATE_HEADERS}
        DESTINATION ${POCL_INSTALL_PRIVATE_HEADER_DIR})

install(FILES "poclu.h"
        DESTINATION ${POCL_INSTALL_PUBLIC_HEADER_DIR})
