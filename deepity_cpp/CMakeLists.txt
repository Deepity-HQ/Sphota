cmake_minimum_required(VERSION 3.21)
project(deepity_local CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(DEEPITY_LOCAL_TSAN "Build with ThreadSanitizer" OFF)

add_library(deepity_local INTERFACE)
target_include_directories(deepity_local INTERFACE include)
target_compile_options(deepity_local INTERFACE -O3 -march=native -fopenmp-simd -Wall -Wextra)
if(DEEPITY_LOCAL_TSAN)
  target_compile_options(deepity_local INTERFACE -fsanitize=thread -g -O1)
  target_link_options(deepity_local INTERFACE -fsanitize=thread)
endif()
find_package(Threads REQUIRED)

foreach(t tLocalLayerGrad tRings)
  add_executable(${t} tests/${t}.cpp)
  target_link_libraries(${t} PRIVATE deepity_local Threads::Threads)
endforeach()
add_executable(bStep bench/bStep.cpp)
target_link_libraries(bStep PRIVATE deepity_local)

add_library(deepity_local_c SHARED src/c_api.cpp)
target_link_libraries(deepity_local_c PRIVATE deepity_local)
