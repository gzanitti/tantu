// RUN: %tantu-opt %s --one-shot-bufferize="bufferize-function-boundaries=true" | FileCheck %s --check-prefix=NORHS

func.func @add_lhs_has_uses(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %sum    = tantu.add %a, %b : tensor<4xf32>
  %double = tantu.add %a, %a : tensor<4xf32>
  return %sum, %double : tensor<4xf32>, tensor<4xf32>
}

// NORHS-NOT: memref.copy
// NORHS-NOT: memref.alloc