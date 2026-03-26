#include "Tantu/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVM/NVVM/Target.h"

namespace mlir::tantu {
#define GEN_PASS_DEF_ATTACHNVVMTARGETPASS
#include "Passes.h.inc"

struct AttachNVVMTargetPass
    : impl::AttachNVVMTargetPassBase<AttachNVVMTargetPass> {
  using AttachNVVMTargetPassBase::AttachNVVMTargetPassBase;
  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    module.walk([&](mlir::gpu::GPUModuleOp gpuModule) {
      auto target = mlir::NVVM::NVVMTargetAttr::get(
          &getContext(), 2, "nvptx64-nvidia-cuda", chip, ptxVersion, nullptr,
          nullptr);
      auto targetsAttr = mlir::ArrayAttr::get(&getContext(), {target});
      gpuModule.setTargetsAttr(targetsAttr);
    });
  }
};

} // namespace mlir::tantu