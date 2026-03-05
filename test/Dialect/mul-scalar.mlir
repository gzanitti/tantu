// RUN: %tantu-opt %s | FileCheck %s

func.func @test_mul_scalar_f32() -> tensor<4xf32> {
  %a = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %s = tantu.constant 3.0 : f32

  // CHECK: tantu.mul_scalar
  %0 = tantu.mul_scalar %a, %s : tensor<4xf32>
  return %0 : tensor<4xf32>
}