cmake_minimum_required(VERSION 3.16)

project(lab LANGUAGES C)

set(CMAKE_C_FLAGS "-march=native -Wall")

find_package(OpenMP REQUIRED)

# set(BLA_VENDOR Intel10_64ilp_seq)
set(BLA_VENDOR OpenBLAS)

find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)

add_subdirectory(mmio)
add_subdirectory(mymath)

add_executable(main main.c)
target_link_libraries(main mymath BLAS::BLAS LAPACK::LAPACK)