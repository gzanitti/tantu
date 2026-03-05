// RUN: %tantu-opt %s --one-shot-bufferize="bufferize-function-boundaries=true" | FileCheck %s --check-prefix=INPLACE

func.func @neg_no_other_uses(%a: tensor<4xf32>) -> tensor<4xf32> {
  %result = tantu.neg %a : tensor<4xf32>
  return %result : tensor<4xf32>
}

// INPLACE-NOT: memref.alloc
// INPLACE-NOT: memref.copy