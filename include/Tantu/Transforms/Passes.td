include "mlir/Pass/PassBase.td"

def TantuToLinalgPass : Pass<"convert-tantu-to-linalg", "mlir::ModuleOp"> {
  let summary = "Lower Tantu dialect to Linalg/Arith/Math/Func";
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::func::FuncDialect",
    "mlir::tensor::TensorDialect",
    "mlir::memref::MemRefDialect"
  ];
}