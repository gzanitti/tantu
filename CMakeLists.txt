cmake_minimum_required(VERSION 3.20)
project(Tantu LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(MLIR REQUIRED CONFIG)

set(TANTU_BINARY_DIR ${PROJECT_BINARY_DIR})
set(TANTU_OPT_PATH ${PROJECT_BINARY_DIR}/tools/tantu-opt/tantu-opt)

message(STATUS "Using MLIR in: ${MLIR_DIR}")
message(STATUS "Using LLVM in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddMLIR)
include(AddLLVM)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

include_directories(${PROJECT_SOURCE_DIR}/examples)

set(LLVM_EXTERNAL_LIT "/usr/local/bin/lit")

enable_testing()

add_subdirectory(include/Tantu)
add_subdirectory(lib/Tantu)
add_subdirectory(tools/tantu-opt)
add_subdirectory(tools/tantu-compiler)
add_subdirectory(test)
add_subdirectory(examples)


# Feature flag for future GPU backend — off by default.
option(TANTU_ENABLE_CUDA "Enable CUDA/GPU backend" OFF)