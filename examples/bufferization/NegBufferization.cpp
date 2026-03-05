// NegBufferization.cpp
//
// External bufferization model for tantu.neg.
//
// Demonstrates how to implement BufferizableOpInterface for a unary
// elementwise op that negates its input operand.

#include "Tantu/BufferizationRegistrations.h"
#include "Tantu/Dialect/TantuDialect.h"
#include "Tantu/Dialect/TantuOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/DialectRegistry.h"

using namespace mlir;
using namespace mlir::bufferization;

namespace {

struct NegBufferizationModel
    : public BufferizableOpInterface::ExternalModel<NegBufferizationModel,
                                                    ::tantu::NegOp> {

  // Does this op read from the input buffer before writing?
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  // Is the input used as a mutable buffer?
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  // Can the result reuse the input buffer?
  // Equivalent = same shape/offset;
  // isDefinite = false lets One-Shot decide.
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getResult(0), BufferRelation::Equivalent,
             /*isDefinite=*/false}};
  }

  // Generate the bufferized version of neg.
  //
  // By the time this method is called, One-Shot has already resolved the
  // in-place decision.
  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto negOp = cast<tantu::NegOp>(op);
    Location loc = op->getLoc();

    // Get the input buffer. May be the original buffer (in-place case) or
    // a fresh copy (non-in-place case).
    FailureOr<Value> inputBuf =
        getBuffer(rewriter, negOp.getOperand(), options);
    if (failed(inputBuf))
      return failure();

    auto memrefType = cast<MemRefType>(inputBuf->getType());
    int64_t rank = memrefType.getRank();

    SmallVector<AffineMap> maps{rewriter.getMultiDimIdentityMap(rank)};
    SmallVector<utils::IteratorType> iterTypes(rank,
                                               utils::IteratorType::parallel);

    rewriter.create<linalg::GenericOp>(
        loc,
        /*inputs=*/ValueRange{},
        /*outputs=*/ValueRange{*inputBuf}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          Value negated = b.create<arith::NegFOp>(loc, args[0]);
          b.create<linalg::YieldOp>(loc, negated);
        });

    replaceOpWithBufferizedValues(rewriter, op, *inputBuf);
    return success();
  }
};

} // namespace

void registerTantuNegBufferizationModel(DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tantu::TantuDialect *dialect) {
    ::tantu::NegOp::attachInterface<NegBufferizationModel>(*ctx);
  });
}