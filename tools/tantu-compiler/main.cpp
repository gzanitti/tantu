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
#ifdef TANTU_ENABLE_CUDA
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/GPUToNVVM/GPUToNVVMPass.h"
#include "mlir/Conversion/NVVMToLLVM/NVVMToLLVM.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#endif

namespace cl = llvm::cl;

static cl::opt<std::string> InputFile(cl::Positional, cl::Required,
                                      cl::desc("<input.tantu>"));
static cl::opt<std::string> OutputFile("o", cl::desc("Output filename"),
                                       cl::value_desc("file"),
                                       cl::init("a.out"));
static cl::opt<bool> EmitMLIR("emit-mlir",
                               cl::desc("Emit Tantu MLIR and stop"));

enum class BackendTarget { CPU, GPU };

static cl::opt<BackendTarget> Target(
    "target", cl::desc("Backend target"),
    cl::values(clEnumValN(BackendTarget::CPU, "cpu", "Lower to CPU (default)"),
               clEnumValN(BackendTarget::GPU, "gpu", "Lower to GPU (naive)")),
    cl::init(BackendTarget::CPU));

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
#ifdef TANTU_ENABLE_CUDA
  mlir::registerNVVMDialectTranslation(registry);
  mlir::registerGPUDialectTranslation(registry);
#endif

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

#ifdef TANTU_ENABLE_CUDA
  if (Target == BackendTarget::GPU) {
    pm.addPass(mlir::createConvertLinalgToParallelLoopsPass());
    pm.nest<mlir::func::FuncOp>().addPass(
        mlir::createGpuMapParallelLoopsPass());
    pm.addPass(mlir::createParallelLoopToGpuPass());
    pm.addPass(mlir::createGpuKernelOutliningPass());
    pm.addPass(mlir::tantu::createAttachNVVMTargetPass());
    {
      auto &gpuPM = pm.nest<mlir::gpu::GPUModuleOp>();
      gpuPM.addPass(mlir::createConvertGpuOpsToNVVMOps());
      gpuPM.addPass(mlir::createLowerAffinePass());
      gpuPM.addPass(mlir::createArithToLLVMConversionPass());
      gpuPM.addPass(mlir::createConvertNVVMToLLVMPass());
      gpuPM.addPass(mlir::createReconcileUnrealizedCastsPass());
    }
    pm.addPass(mlir::createGpuModuleToBinaryPass());
  } else
#endif
  {
    pm.addPass(mlir::createConvertLinalgToAffineLoopsPass());
    pm.addPass(mlir::createLowerAffinePass());
    pm.addPass(mlir::createConvertSCFToCFPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  }
  if (mlir::failed(pm.run(module))) {
    llvm::WithColor::error() << "lowering failed\n";
    return 1;
  }

#ifdef TANTU_ENABLE_CUDA
  if (Target == BackendTarget::GPU) {
    mlir::gpu::BinaryOp binary;
    module.walk([&](mlir::gpu::BinaryOp op) { binary = op; });
    if (!binary) {
      llvm::WithColor::error() << "no gpu.binary found in module\n";
      return 1;
    }

    auto objects = binary.getObjects();
    if (objects.empty()) {
      llvm::WithColor::error() << "gpu.binary has no objects\n";
      return 1;
    }
    auto object = mlir::cast<mlir::gpu::ObjectAttr>(objects[0]);
    llvm::StringRef cubinBytes = object.getObject().getValue();

    std::error_code ec;
    llvm::raw_fd_ostream out(OutputFile, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::WithColor::error()
          << "cannot open output file: " << ec.message() << "\n";
      return 1;
    }
    out.write(cubinBytes.data(), cubinBytes.size());
    return 0;
  }
#endif

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

  auto tm = target->createTargetMachine(targetTriple, "generic", "", {}, {});
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
