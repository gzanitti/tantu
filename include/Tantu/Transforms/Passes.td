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

def LowerTantuPrintPass : Pass<"lower-tantu-print", "mlir::ModuleOp"> {
  let summary = "Lower tantu.print to printMemrefF32 runtime call";
  let dependentDialects = [
    "mlir::func::FuncDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def FuseElementwisePass : Pass<"fuse-elementwise", "mlir::ModuleOp"> {
  let summary = "Fuse elementwise operations into a single operation";
}

def AttachNVVMTargetPass : Pass<"attach-nvvm-target", "mlir::ModuleOp"> {
  let summary = "Attach NVVM target attribute to gpu.module";
  let options = [
    Option<"chip", "chip", "std::string",
           "\"sm_61\"", "Target GPU architecture (e.g. sm_61, sm_80)">,
    Option<"ptxVersion", "ptx-version", "std::string",
           "\"+ptx61\"", "PTX feature string (e.g. +ptx61, +ptx80)">
  ];
  let dependentDialects = [
    "mlir::gpu::GPUDialect",
    "mlir::NVVM::NVVMDialect"
  ];
}