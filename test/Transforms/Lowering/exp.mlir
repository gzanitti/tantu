// RUN: %tantu-opt --convert-tantu-to-linalg %s | FileCheck %s

func.func @test_exp(%a: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = tantu.exp %a : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_exp
// CHECK: linalg.generic
// CHECK: math.exp
// CHECK-NOT: tantu.exp