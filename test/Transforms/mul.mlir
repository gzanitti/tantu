// RUN: %tantu-opt --convert-tantu-to-linalg %s | FileCheck %s

func.func @test_mul(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = tantu.mul %a, %b : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_mul
// CHECK: linalg.generic
// CHECK: arith.mulf
// CHECK-NOT: tantu.mul