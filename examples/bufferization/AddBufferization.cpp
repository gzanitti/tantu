// AddBufferization.cpp
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

struct AddBufferizationModel
    : public BufferizableOpInterface::ExternalModel<AddBufferizationModel,
                                                    tantu::AddOp> {

  // Does this op read from the input buffer before writing?
  // False only for ops that overwrite without reading (e.g. fill).
  bool bufferizesToMemoryRead(Operation *op, OpOperand &opOperand,
                              const AnalysisState &state) const {
    return true;
  }

  // Is the input operand a mutable in-out buffer?
  // For aliasing potential, see getAliasingValues.
  bool bufferizesToMemoryWrite(Operation *op, OpOperand &opOperand,
                               const AnalysisState &state) const {
    return false;
  }

  // Can the result reuse the input buffer?
  // Called once per operand — we declare the result as a candidate alias
  // for each. One-Shot picks whichever operand has no further uses.
  AliasingValueList getAliasingValues(Operation *op, OpOperand &opOperand,
                                      const AnalysisState &state) const {
    return {{op->getResult(0), BufferRelation::Equivalent,
             /*isDefinite=*/false}};
  }

  LogicalResult bufferize(Operation *op, RewriterBase &rewriter,
                          const BufferizationOptions &options) const {
    auto addOp = cast<tantu::AddOp>(op);
    Location loc = op->getLoc();

    // Get input buffers for both operands.
    FailureOr<Value> lhsBuf = getBuffer(rewriter, addOp.getLhs(), options);
    if (failed(lhsBuf))
      return failure();

    FailureOr<Value> rhsBuf = getBuffer(rewriter, addOp.getRhs(), options);
    if (failed(rhsBuf))
      return failure();

    // The output buffer may alias either lhs or rhs if One-Shot decided
    // in-place. We use lhs as the base for type/rank information.
    auto memrefType = cast<MemRefType>(lhsBuf->getType());
    int64_t rank = memrefType.getRank();

    // For binary elementwise ops we need a separate output buffer.
    // One-Shot may have made it alias lhs or rhs, but we receive it
    // via getBuffer on the result — which handles that transparently.
    FailureOr<Value> outBuf = getBuffer(rewriter, addOp.getLhs(), options);
    if (failed(outBuf))
      return failure();

    // Identity map for each of the three operands: lhs, rhs, out.
    SmallVector<AffineMap> maps{rewriter.getMultiDimIdentityMap(rank),  // lhs
                                rewriter.getMultiDimIdentityMap(rank),  // rhs
                                rewriter.getMultiDimIdentityMap(rank)}; // out

    SmallVector<utils::IteratorType> iterTypes(rank,
                                               utils::IteratorType::parallel);

    rewriter.create<linalg::GenericOp>(
        loc,
        /*inputs=*/ValueRange{*lhsBuf, *rhsBuf},
        /*outputs=*/ValueRange{*outBuf}, maps, iterTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = lhs element, args[1] = rhs element, args[2] = out
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    replaceOpWithBufferizedValues(rewriter, op, *outBuf);
    return success();
  }
};

} // namespace

void registerTantuAddBufferizationModel(mlir::DialectRegistry &registry) {
  registry.addExtension(+[](MLIRContext *ctx, tantu::TantuDialect *dialect) {
    tantu::AddOp::attachInterface<AddBufferizationModel>(*ctx);
  });
}