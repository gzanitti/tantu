// RUN: %tantu-opt --convert-tantu-to-linalg %s | FileCheck %s

func.func @test_neg(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = tantu.neg %a : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_neg
// CHECK: linalg.generic
// CHECK: arith.negf
// CHECK-NOT: tantu.neg