cmake_minimum_required(VERSION 3.16)

project(MPNA LANGUAGES C)

# Set C standard to 11
set(CMAKE_C_STANDARD 11)

# Set compiler flags
include(CheckCCompilerFlag)

# Enable all warnings
set(CMAKE_C_FLAGS "-Wall -Wextra -Wpedantic")

check_c_compiler_flag(-march=native HAS_MARCH_NATIVE)

if (HAS_MARCH_NATIVE)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
endif()


add_subdirectory(src)

