#include "Tantu/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tantu {

#define GEN_PASS_DEF_FUSEELEMENTWISEPASS
#include "FuseElementwise.h.inc"
#include "Passes.h.inc"

static LogicalResult isElementwise(PatternRewriter &rewriter, Operation *op) {
  auto genericOp = llvm::dyn_cast<linalg::GenericOp>(op);
  if (!genericOp)
    return failure();
  auto attrs = genericOp.getIteratorTypesArray();
  for (size_t i = 0; i < attrs.size(); i++) {
    if (attrs[i] != utils::IteratorType::parallel)
      return failure();
  }

  return success();
}

static Operation *fuseGenerics(PatternRewriter &rewriter, Operation *op1,
                               Operation *op2) {
  auto generic1 = llvm::cast<linalg::GenericOp>(op1);
  auto generic2 = llvm::cast<linalg::GenericOp>(op2);

  SmallVector<AffineMap> mapsVec;
  auto op1Maps = generic1.getIndexingMapsArray();
  mapsVec.push_back(op1Maps[0]);
  mapsVec.push_back(op1Maps[1]);
  auto op2Maps = generic2.getIndexingMapsArray();
  mapsVec.push_back(op2Maps.back());

  auto loc = op1->getLoc();
  auto resultType = op2->getResultTypes();
  auto iterators = generic1.getIteratorTypesArray();

  auto fusedOp = rewriter.create<linalg::GenericOp>(
      loc, resultType,
      ValueRange{generic1->getOperand(0), generic1->getOperand(1)},
      ValueRange{generic2.getOutputs()}, mapsVec, iterators);

  Block *block1 = generic1.getBody();
  Block *block2 = generic2.getBody();
  Region &region = fusedOp.getRegion();
  Block *fusedBlock = &region.emplaceBlock();

  auto b1Arg1 = block1->getArgument(0);
  auto b1Arg2 = block1->getArgument(1);
  auto b2Arg = block2->getArguments().back();

  fusedBlock->addArgument(b1Arg1.getType(), b1Arg1.getLoc());
  fusedBlock->addArgument(b1Arg2.getType(), b1Arg2.getLoc());
  fusedBlock->addArgument(b2Arg.getType(), b2Arg.getLoc());

  IRMapping mapping1;
  mapping1.map(block1->getArgument(0), fusedBlock->getArgument(0));
  mapping1.map(block1->getArgument(1), fusedBlock->getArgument(1));

  rewriter.setInsertionPointToStart(fusedBlock);
  for (auto &op : block1->without_terminator())
    rewriter.clone(op, mapping1);

  auto yield1 = cast<linalg::YieldOp>(block1->getTerminator());
  Value yieldedValue = mapping1.lookup(yield1->getOperand(0));

  IRMapping mapping2;
  mapping2.map(block2->getArgument(0), yieldedValue);
  mapping2.map(block2->getArguments().back(),
               fusedBlock->getArguments().back());

  for (auto &op : *block2)
    rewriter.clone(op, mapping2);

  return fusedOp;
}

struct FuseElementwisePass
    : impl::FuseElementwisePassBase<FuseElementwisePass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    populateGeneratedPDLLPatterns(patterns);
    patterns.getPDLPatterns().registerConstraintFunction("isElementwise",
                                                         isElementwise);
    patterns.getPDLPatterns().registerRewriteFunction("fuseGenerics",
                                                      fuseGenerics);
    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::tantu