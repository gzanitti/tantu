#ifndef TANTU_OPS_H
#define TANTU_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Tantu/Dialect/TantuDialect.h"

#define GET_OP_CLASSES
#include "Tantu/Dialect/TantuOps.h.inc"

#endif