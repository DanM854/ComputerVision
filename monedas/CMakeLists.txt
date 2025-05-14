cmake_minimum_required(VERSION 2.8)

project(monedas)

include_directories("/usr/local/include")
find_package( OpenCV REQUIRED )

add_executable(${PROJECT_NAME} "main.cpp")

target_link_libraries( ${PROJECT_NAME}  ${OpenCV_LIBS} )
