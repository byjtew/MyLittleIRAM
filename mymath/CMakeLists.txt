add_library(mymath
        mymath.c mymath.h
        )

target_link_libraries(mymath m OpenMP::OpenMP_C BLAS::BLAS LAPACK::LAPACK)
target_include_directories(mymath PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})