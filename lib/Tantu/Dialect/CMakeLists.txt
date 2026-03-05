add_mlir_library(TantuDialect
  TantuDialect.cpp
  TantuOps.cpp

  DEPENDS
    TantuOpsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRInferTypeOpInterface
)