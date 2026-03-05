set(LLVM_TARGET_DEFINITIONS
  "${PROJECT_SOURCE_DIR}/include/Tantu/Transforms/Passes.td")

mlir_tablegen(Passes.h.inc -gen-pass-decls
  -name Tantu
)
add_public_tablegen_target(TantuTransformsIncGen)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_library(TantuTransforms
  LowerToLinalg.cpp
  LowerTantuPrint.cpp
  DEPENDS
    TantuTransformsIncGen
  LINK_LIBS PUBLIC
    TantuDialect
    MLIRLinalgDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRMemRefDialect
    MLIRTransforms
)