
add_executable(mandelbrot src/main.cpp)
target_link_libraries (mandelbrot PRIVATE ${SYCL_LIBRARIES})

add_executable(mandelbrot_usm src/main.cpp)
target_compile_definitions(mandelbrot_usm PRIVATE MANDELBROT_USM)
target_link_libraries (mandelbrot_usm PRIVATE ${SYCL_LIBRARIES})

add_test (NAME oneapi_mandelbrot
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/mandelbrot
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

add_test (NAME oneapi_mandelbrot_usm
        COMMAND ${CMAKE_CURRENT_BINARY_DIR}/mandelbrot_usm
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})


