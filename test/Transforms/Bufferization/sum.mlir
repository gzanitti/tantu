// RUN: %tantu-opt --convert-tantu-to-linalg --one-shot-bufferize="bufferize-function-boundaries=true" %s | FileCheck %s

func.func @test_sum_bufferized(%a: tensor<4x4xf32>) -> f32 {
  %0 = tantu.sum %a : tensor<4x4xf32> -> f32
  return %0 : f32
}

// CHECK-LABEL: func.func @test_sum_bufferized
// CHECK: memref.alloc
// CHECK: linalg.fill
// CHECK: linalg.reduce
// CHECK: arith.addf
// CHECK: memref.load
// CHECK-NOT: tensor
