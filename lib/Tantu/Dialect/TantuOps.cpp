#include "Tantu/Dialect/TantuOps.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallSet.h"

using namespace ::mlir;
using namespace ::tantu;
using namespace mlir::bufferization;

#define GET_OP_CLASSES
#include "Tantu/Dialect/TantuOps.cpp.inc"

::mlir::LogicalResult ConstantOp::verify() {
  auto attrType = getValue().getType();

  if (!attrType.isF32() && !attrType.isInteger(32))
    return emitOpError("unsupported constant type ")
           << attrType << "; expected f32 or i32";

  return success();
}

::mlir::LogicalResult TransposeOp::verify() {
  auto inputType = llvm::cast<RankedTensorType>(getInput().getType());
  auto perm = getPerm();
  auto resultType = getResult().getType();

  if (perm.size() != inputType.getRank())
    return emitOpError("unsupported permutation size ")
           << perm.size() << " and " << inputType.getRank();

  llvm::SmallSet<int64_t, 8> seen;
  for (int64_t i = 0; i < (int64_t)perm.size(); i++) {
    int64_t val = llvm::cast<IntegerAttr>(perm[i]).getInt();
    if (val < 0 || val >= inputType.getRank())
      return emitOpError("perm value out of range: ") << val;
    if (!seen.insert(val).second)
      return emitOpError("perm contains duplicate value: ") << val;
    if (inputType.getShape()[val] != resultType.getShape()[i])
      return emitOpError("result shape mismatch at dimension ") << i;
  }

  return success();
}

::mlir::LogicalResult MatmulOp::verify() {
  auto lhsType = llvm::cast<RankedTensorType>(getOperand(0).getType());
  auto rhsType = llvm::cast<RankedTensorType>(getOperand(1).getType());

  auto resultType = getResult().getType();

  if (lhsType.getRank() != 2 || rhsType.getRank() != 2)
    return emitOpError("unsupported tensor rank ")
           << lhsType.getRank() << " or " << rhsType.getRank();

  if (lhsType.getShape()[1] != rhsType.getShape()[0])
    return emitOpError("unsupported tensor shape ")
           << lhsType.getShape()[1] << " and " << rhsType.getShape()[0];

  if (lhsType.getShape()[0] != resultType.getShape()[0] ||
      rhsType.getShape()[1] != resultType.getShape()[1])
    return emitOpError("inner dimensions must match for matmul: ")
           << lhsType.getShape() << " and " << rhsType.getShape() << " -> "
           << resultType.getShape();

  return success();
}
bool PrintOp::bufferizesToMemoryRead(OpOperand &opOperand,
                                     const AnalysisState &state) {
  return true;
}

bool PrintOp::bufferizesToMemoryWrite(OpOperand &opOperand,
                                      const AnalysisState &state) {
  return false;
}

AliasingValueList PrintOp::getAliasingValues(OpOperand &opOperand,
                                             const AnalysisState &state) {
  return {};
}

LogicalResult PrintOp::bufferize(RewriterBase &rewriter,
                                 const BufferizationOptions &options) {
  FailureOr<Value> inputBuf = getBuffer(rewriter, getInput(), options);
  if (failed(inputBuf))
    return failure();

  rewriter.replaceOpWithNewOp<PrintOp>(getOperation(), *inputBuf);
  return success();
}