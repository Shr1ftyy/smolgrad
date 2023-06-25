cmake_minimum_required(VERSION 3.10)

# Set the project name
project(NN)

# Set the C Standard
set(CMAKE_C_STANDARD 99)

# Include directories
include_directories(src include)


# Add the executable
add_executable(unit_test
  tests/unit_test.c
)

# add_executable(test_map
#   tests/test_map.c
# )

add_executable(main_test
  tests/main_test.c
)

target_link_libraries(unit_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/unity/unity.c)
target_link_libraries(unit_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/hashmap/hashmap.c)
# target_link_libraries(test_map PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/hashmap/hashmap.c)
target_link_libraries(main_test PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/hashmap/hashmap.c)

# Set debug flags
# Specify the directory for the binary output
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ansi -Wall -DUNITY_INCLUDE_DOUBLE -DUNITY_DOUBLE_PRECISION=1e-12")
