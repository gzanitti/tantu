// RUN: %tantu-opt --convert-tantu-to-linalg %s | FileCheck %s

func.func @test_sum(%a: tensor<4x4xf32>) -> f32 {
  %0 = tantu.sum %a : tensor<4x4xf32> -> f32
  return %0 : f32
}

// CHECK-LABEL: func.func @test_sum
// CHECK: tensor.empty
// CHECK: linalg.fill
// CHECK: linalg.reduce
// CHECK: arith.addf
// CHECK: tensor.extract
// CHECK-NOT: tantu.sum