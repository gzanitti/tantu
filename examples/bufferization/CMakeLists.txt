add_mlir_library(TantuBufferizationExamples
  NegBufferization.cpp
  AddBufferization.cpp

  LINK_LIBS PUBLIC
  TantuDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
)

add_subdirectory(test)