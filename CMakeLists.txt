cmake_minimum_required(VERSION 3.10.1)

project(cgrad)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(${PROJECT_NAME} src/main.cpp)

include_directories(inc/)

add_library(main src/main.cpp)
