#include "Tantu/BufferizationRegistrations.h"
#include "Tantu/Dialect/TantuDialect.h"
#include "Tantu/Dialect/TantuOps.h"
#include "Tantu/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<tantu::TantuDialect>();

  registerTantuNegBufferizationModel(registry);
  registerTantuAddBufferizationModel(registry);

  mlir::registerAllPasses();
  mlir::tantu::registerTantuPasses();

  mlir::PassPipelineRegistration<>(
      "lower-to-cpu", "Full lowering pipeline to LLVM IR for CPU",
      [](mlir::OpPassManager &pm) {
        pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createConvertSCFToCFPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
      });

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tantu optimizer driver\n", registry));
}
