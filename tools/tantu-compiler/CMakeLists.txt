add_executable(tantu-compiler main.cpp)

target_include_directories(tantu-compiler PRIVATE
  ${CMAKE_BINARY_DIR}/lib/Tantu/Transforms
)

target_link_libraries(tantu-compiler
  PRIVATE
  TantuFrontend
  TantuSema
  TantuIRGen
  TantuDialect
  TantuTransforms
  TantuBufferizationExamples
  MLIR
)

if(TANTU_ENABLE_CUDA)
  target_compile_definitions(tantu-compiler PRIVATE TANTU_ENABLE_CUDA)
  target_link_libraries(tantu-compiler PRIVATE
    MLIRGPUDialect
    MLIRGPUTransforms
    MLIRNVVMDialect
    MLIRNVVMTarget
    MLIRSCFToGPU
  )
endif()