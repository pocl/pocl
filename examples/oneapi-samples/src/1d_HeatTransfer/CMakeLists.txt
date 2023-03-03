add_executable (1d_HeatTransfer src/1d_HeatTransfer.cpp)
target_link_libraries (1d_HeatTransfer PRIVATE ${SYCL_LIBRARIES})

add_test (NAME oneapi_1d_HeatTransfer
	COMMAND ${CMAKE_CURRENT_BINARY_DIR}/1d_HeatTransfer 100 1000
	WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
