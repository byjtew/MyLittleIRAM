

add_subdirectory(mmio)
add_subdirectory(mymath)

add_executable(main main.c)
target_link_libraries(main mymath)
set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)