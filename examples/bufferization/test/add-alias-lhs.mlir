// RUN: %tantu-opt %s --one-shot-bufferize="bufferize-function-boundaries=true" | FileCheck %s --check-prefix=NOLHS

func.func @add_rhs_has_uses(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %sum    = tantu.add %a, %b : tensor<4xf32>
  %double = tantu.add %b, %b : tensor<4xf32>
  return %sum, %double : tensor<4xf32>, tensor<4xf32>
}

// NOLHS-NOT: memref.copy
// NOLHS-NOT: memref.alloc