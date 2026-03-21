#include "Tantu/Dialect/TantuDialect.h"
#include "Tantu/Dialect/TantuOps.h"

using namespace ::mlir;
using namespace ::tantu;

#include "Tantu/TantuDialect.cpp.inc"

void TantuDialect::initialize() {
  addOperations<ConstantOp, TensorLiteralOp, AddOp, SubOp, MulOp, DivOp, MaxOp,
                ExpOp, NegOp, AddScalarOp, SubScalarOp, MulScalarOp,
                DivScalarOp, MaxScalarOp, SumOp, MaxReduceOp, SizeOp, CastOp,
                TransposeOp, MatmulOp, PrintOp>();
}