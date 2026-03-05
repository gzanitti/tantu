#pragma once
#include "mlir/Pass/Pass.h"

namespace mlir::tantu {

#define GEN_PASS_DECL
#include "Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

} // namespace mlir::tantu