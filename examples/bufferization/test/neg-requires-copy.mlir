// RUN: %tantu-opt %s --one-shot-bufferize | FileCheck %s --check-prefix=COPY

func.func @neg_with_other_use(%a: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>) {
  %neg    = tantu.neg %a : tensor<4xf32>
  %double = tantu.add %a, %a : tensor<4xf32>
  return %neg, %double : tensor<4xf32>, tensor<4xf32>
}

// COPY: memref.alloc