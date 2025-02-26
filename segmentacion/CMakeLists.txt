cmake_minimum_required(VERSION 2.8) 
project(segmentacion)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(segmentacion main.cpp)

target_link_libraries(segmentacion ${OpenCV_LIBS})
