cmake_minimum_required (VERSION 3.5.0)
project (HELLO)
set(CMAKE_BUILD_TYPE Debug)
#find_package(libyaml REQUIRED)
add_executable (helloDemo demo.cpp gstaction.cpp)
#include_directories(${YAML_CPP_INCLUDE_DIRS})
target_link_libraries(helloDemo yaml)

