set(LLVM_TARGET_DEFINITIONS
  "${PROJECT_SOURCE_DIR}/include/Tantu/Transforms/Passes.td")

mlir_tablegen(Passes.h.inc -gen-pass-decls
  -name Tantu
)
add_public_tablegen_target(TantuTransformsIncGen)

add_mlir_pdll_library(TantuFuseElementwiseIncGen FuseElementwise.pdll FuseElementwise.h.inc)


include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_library(TantuTransforms
  LowerToLinalg.cpp
  LowerTantuPrint.cpp
  FuseElementwise.cpp
  DEPENDS
    TantuTransformsIncGen
    TantuFuseElementwiseIncGen
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