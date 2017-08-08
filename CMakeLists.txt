cmake_minimum_required(VERSION 2.8.11)
project(halfsize_sse_benchmark CXX C)

set(CMAKE_CXX_STANDARD 11)

if(MSVC)
  if(ARCH_32)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:SSE2")
  endif()
    add_definitions(-D__SSE__)
else()
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif()

add_executable(halfsize_sse_benchmark main.cpp)

enable_testing()
add_test(halfsize_sse_benchmark halfsize_sse_benchmark)
