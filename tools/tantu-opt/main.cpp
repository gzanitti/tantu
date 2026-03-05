#include "Tantu/BufferizationRegistrations.h"
#include "Tantu/Dialect/TantuDialect.h"
#include "Tantu/Dialect/TantuOps.h"
#include "Tantu/Transforms/Passes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<tantu::TantuDialect>();

  registerTantuNegBufferizationModel(registry);
  registerTantuAddBufferizationModel(registry);

  mlir::registerAllPasses();
  mlir::tantu::registerTantuPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Tantu optimizer driver\n", registry));
}