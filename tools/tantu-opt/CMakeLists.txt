add_executable(tantu-opt main.cpp)

target_include_directories(tantu-opt PRIVATE
  ${CMAKE_BINARY_DIR}/lib/Tantu/Transforms
)

target_link_libraries(tantu-opt
  PRIVATE
  TantuDialect
  TantuTransforms
  TantuBufferizationExamples
  MLIROptLib
  MLIR
)