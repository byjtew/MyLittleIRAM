

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

message(STATUS "Checking for BLAS and LAPACK")
option(BLA_VENDOR "BLAS vendor" "OpenBLAS")

if (BLA_VENDOR)
  message(STATUS "BLAS vendor: ${BLA_VENDOR}")
endif()
find_package(LAPACK REQUIRED)
find_package(BLAS REQUIRED)


add_library(mymath STATIC
        mymath.c mymath.h
)
target_link_libraries(mymath m OpenMP::OpenMP_C BLAS::BLAS LAPACK::LAPACK MPI::MPI_C)
target_include_directories(mymath PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(mymath PROPERTIES ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)