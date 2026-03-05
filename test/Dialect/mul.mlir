// RUN: %tantu-opt %s | FileCheck %s

func.func @test_mul_f32() -> tensor<4xf32> {
  %a = arith.constant dense<[2.0, 3.0, 4.0, 5.0]> : tensor<4xf32>
  %b = arith.constant dense<[0.5, 2.0, 0.25, 4.0]> : tensor<4xf32>

  // CHECK: tantu.mul
  %0 = tantu.mul %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}