// RUN: %tantu-opt %s --one-shot-bufferize | FileCheck %s --check-prefix=ALLOC

func.func @add_both_have_uses(%a: tensor<4xf32>, %b: tensor<4xf32>) -> (tensor<4xf32>, tensor<4xf32>, tensor<4xf32>) {
  %sum  = tantu.add %a, %b : tensor<4xf32>
  %nega = tantu.neg %a : tensor<4xf32>
  %negb = tantu.neg %b : tensor<4xf32>
  return %sum, %nega, %negb : tensor<4xf32>, tensor<4xf32>, tensor<4xf32>
}

// ALLOC: memref.alloc