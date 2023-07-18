
add_executable(simple-add src/simple-add-buffers.cpp)
target_link_libraries (simple-add PRIVATE ${SYCL_LIBRARIES})

add_executable(simple-add_usm src/simple-add-usm.cpp)
target_link_libraries (simple-add_usm PRIVATE ${SYCL_LIBRARIES})

add_test (NAME oneapi_simple_add
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/simple-add
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test (NAME oneapi_simple_add_usm
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/simple-add_usm
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

