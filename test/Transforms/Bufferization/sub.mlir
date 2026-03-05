// RUN: %tantu-opt --convert-tantu-to-linalg --one-shot-bufferize="bufferize-function-boundaries=true" %s | FileCheck %s

func.func @test_sub_bufferized(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = tantu.sub %a, %b : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_sub_bufferized
// CHECK: memref.alloc
// CHECK: linalg.generic
// CHECK: arith.subf
// CHECK-NOT: tensor
