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
