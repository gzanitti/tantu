// RUN: %tantu-opt --convert-tantu-to-linalg %s | FileCheck %s

func.func @test_add(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = tantu.add %a, %b : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_add
// CHECK: linalg.generic
// CHECK: arith.addf
// CHECK-NOT: tantu.add