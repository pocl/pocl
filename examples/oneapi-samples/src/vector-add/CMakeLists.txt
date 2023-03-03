
add_executable(vector-add src/vector-add-buffers.cpp)
target_link_libraries (vector-add PRIVATE ${SYCL_LIBRARIES})

add_executable(vector-add_usm src/vector-add-usm.cpp)
target_link_libraries (vector-add_usm PRIVATE ${SYCL_LIBRARIES})

add_test (NAME oneapi_vector_add
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vector-add
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test (NAME oneapi_vector_add_usm
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/vector-add_usm
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

