#include "Tantu/Dialect/TantuDialect.h"
#include "Tantu/Dialect/TantuOps.h"
#include "Tantu/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tantu {

#define GEN_PASS_DEF_LOWERTANTUPRINTPASS
#include "Passes.h.inc"

struct PrintOpLowering : public OpRewritePattern<::tantu::PrintOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(::tantu::PrintOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    auto module = op->getParentOfType<ModuleOp>();
    if (module.lookupSymbol<func::FuncOp>("printMemrefF32") == nullptr) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(module.getBody());

      auto unrankedType = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
      auto funcType =
          FunctionType::get(rewriter.getContext(), {unrankedType}, {});

      auto funcOp =
          rewriter.create<func::FuncOp>(loc, "printMemrefF32", funcType);
      funcOp.setPrivate();
    }

    auto unrankedType = UnrankedMemRefType::get(rewriter.getF32Type(), 0);
    auto castedBuf =
        rewriter.create<memref::CastOp>(loc, unrankedType, op.getInput());

    rewriter.create<func::CallOp>(loc, "printMemrefF32", TypeRange{},
                                  castedBuf.getResult());
    rewriter.eraseOp(op);

    return success();
  }
};

struct LowerTantuPrintPass
    : public impl::LowerTantuPrintPassBase<LowerTantuPrintPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.add<PrintOpLowering>(ctx);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::tantu