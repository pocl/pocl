
add_executable (nbody src/GSimulation.cpp src/main.cpp)
target_link_libraries (nbody PRIVATE ${SYCL_LIBRARIES})

add_test (NAME oneapi_nbody
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/nbody
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
