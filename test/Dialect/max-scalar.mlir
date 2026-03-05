// RUN: %tantu-opt %s | FileCheck %s

func.func @test_max_scalar_f32() -> tensor<4xf32> {
  %a = arith.constant dense<[-1.0, 2.0, -3.0, 4.0]> : tensor<4xf32>
  %s = tantu.constant 0.0 : f32

  // CHECK: tantu.max_scalar
  %0 = tantu.max_scalar %a, %s : tensor<4xf32>
  return %0 : tensor<4xf32>
}