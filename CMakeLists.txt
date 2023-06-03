cmake_minimum_required(VERSION 3.10)

# Set the project name
project(NN)

# Set the C Standard
set(CMAKE_C_STANDARD 99)

# Set the default build type to Debug
# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Build CACHE STRING "Choose the type of build (Build)" FORCE)
# else()
#   set(CMAKE_C_FLAGS_DEBUG "-g -DDEBUG")
# endif()

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the executable
add_executable(${PROJECT_NAME} 
  src/main.c
)

add_executable(tests 
  tests/test.c
)

# Set debug flags
# Specify the directory for the binary output
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)