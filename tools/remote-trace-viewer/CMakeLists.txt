cmake_minimum_required(VERSION 3.31)
project(TraceViewer CXX)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(SDL3 REQUIRED CONFIG REQUIRED COMPONENTS SDL3 Headers)

add_executable(viewer trace-viewer.cc)
set_property(TARGET viewer PROPERTY CXX_STANDARD 17)
set_property(TARGET viewer PROPERTY CXX_STANDARD_REQUIRED ON)
set_property(TARGET viewer PROPERTY CXX_EXTENSIONS OFF)
target_link_libraries(viewer PUBLIC SDL3::SDL3)
