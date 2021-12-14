cmake_minimum_required (VERSION 3.5.0)
project (HELLO)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_BUILD_TYPE Debug)
#find_package(libyaml REQUIRED)
add_executable (helloDemo src/node_yaml.cpp src/yaml_utils.cpp)
include_directories(include)
target_link_libraries(helloDemo yaml)

