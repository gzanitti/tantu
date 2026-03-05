// RUN: %tantu-opt --convert-tantu-to-linalg %s | FileCheck %s

func.func @test_max_scalar(%a: tensor<4x4xf32>, %s: f32) -> tensor<4x4xf32> {
  %0 = tantu.max_scalar %a, %s : tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_max_scalar
// CHECK: tensor.splat
// CHECK: linalg.generic
// CHECK: arith.maximumf
// CHECK-NOT: tantu.max_scalar