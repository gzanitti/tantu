#include "Tantu/Dialect/TantuDialect.h"
#include "Tantu/Frontend/Lexer.h"
#include "Tantu/Frontend/Parser.h"
#include "Tantu/IRGen/IRGen.h"
#include "Tantu/Sema/TypeChecker.h"
#include "Tantu/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/WithColor.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"

namespace cl = llvm::cl;

static cl::opt<std::string> InputFile(cl::Positional, cl::Required,
                                      cl::desc("<input.tantu>"));
static cl::opt<std::string> OutputFile("o", cl::desc("Output filename"),
                                       cl::value_desc("file"),
                                       cl::init("a.out"));
static cl::opt<bool> EmitMLIR("emit-mlir",
                               cl::desc("Emit Tantu MLIR and stop"));

int main(int argc, char **argv) {
  cl::ParseCommandLineOptions(argc, argv, "Tantu compiler\n");

  auto bufOrErr = llvm::MemoryBuffer::getFile(InputFile);
  if (!bufOrErr) {
    llvm::WithColor::error() << InputFile << ": "
                             << bufOrErr.getError().message() << "\n";
    return 1;
  }
  std::string source = (*bufOrErr)->getBuffer().str();

  Lexer lexer(source);
  Parser parser(std::move(lexer));
  auto progOrErr = parser.parseProgram();
  if (!progOrErr) {
    llvm::handleAllErrors(progOrErr.takeError(),
                          [](const llvm::StringError &e) {
                            llvm::WithColor::error() << e.getMessage() << "\n";
                          });
    return 1;
  }

  TypeChecker tc;
  if (llvm::Error err = tc.check((*progOrErr).get())) {
    llvm::handleAllErrors(std::move(err), [](const llvm::StringError &e) {
      llvm::WithColor::error() << e.getMessage() << "\n";
    });
    return 1;
  }

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<tantu::TantuDialect>();
  mlir::registerLLVMDialectTranslation(registry);

  mlir::MLIRContext ctx(registry);
  ctx.loadAllAvailableDialects();

  IRGen irgen(&ctx);
  irgen.emit(*(*progOrErr));
  mlir::ModuleOp module = irgen.getModule();

  if (EmitMLIR) {
    module.print(llvm::outs());
    return 0;
  }

  mlir::PassManager pm(&ctx);
  pm.addPass(mlir::tantu::createTantuToLinalgPass());
  mlir::bufferization::OneShotBufferizationOptions bufOpts;
  bufOpts.bufferizeFunctionBoundaries = true;
  pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufOpts));
  pm.addPass(mlir::tantu::createLowerTantuPrintPass());
  pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());

  if (mlir::failed(pm.run(module))) {
    llvm::WithColor::error() << "lowering failed\n";
    return 1;
  }

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  llvm::LLVMContext llvmCtx;
  auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmCtx);
  if (!llvmModule) {
    llvm::WithColor::error() << "LLVM IR translation failed\n";
    return 1;
  }

  std::string targetTriple = llvm::sys::getDefaultTargetTriple();
  llvmModule->setTargetTriple(targetTriple);

  std::string err;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(targetTriple, err);
  if (!target) {
    llvm::WithColor::error() << err << "\n";
    return 1;
  }

  auto tm =
      target->createTargetMachine(targetTriple, "generic", "", {}, {});
  llvmModule->setDataLayout(tm->createDataLayout());

  llvm::SmallString<128> objPath;
  {
    int fd;
    if (llvm::sys::fs::createTemporaryFile("tantu", "o", fd, objPath)) {
      llvm::WithColor::error() << "failed to create temporary object file\n";
      return 1;
    }
    llvm::raw_fd_ostream objStream(fd, /*shouldClose=*/true);
    llvm::legacy::PassManager codegenPM;
    if (tm->addPassesToEmitFile(codegenPM, objStream, nullptr,
                                llvm::CodeGenFileType::ObjectFile)) {
      llvm::WithColor::error() << "target does not support object emission\n";
      return 1;
    }
    codegenPM.run(*llvmModule);
  }

  auto ccOrErr = llvm::sys::findProgramByName("cc");
  if (!ccOrErr) {
    llvm::WithColor::error() << "cc not found\n";
    llvm::sys::fs::remove(objPath);
    return 1;
  }
  std::string ccPath = *ccOrErr;
  std::vector<llvm::StringRef> linkArgs = {ccPath, objPath, "-o", OutputFile};
  int ret = llvm::sys::ExecuteAndWait(ccPath, linkArgs);
  llvm::sys::fs::remove(objPath);
  return ret;
}
