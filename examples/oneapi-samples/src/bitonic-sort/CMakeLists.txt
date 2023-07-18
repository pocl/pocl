add_executable (bitonic-sort src/bitonic-sort.cpp)
target_link_libraries (bitonic-sort PRIVATE ${SYCL_LIBRARIES})

add_test (NAME oneapi_bitonic_sort
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/bitonic-sort 21 47
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
