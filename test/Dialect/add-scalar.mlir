// RUN: %tantu-opt %s | FileCheck %s

func.func @test_add_scalar_f32() -> tensor<4xf32> {
  %a = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %b = tantu.constant 1.0 : f32

  // CHECK: tantu.add_scalar
  %0 = tantu.add_scalar %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}