// RUN: %tantu-opt %s | FileCheck %s

func.func @test_div_scalar_f32() -> tensor<4xf32> {
  %a = arith.constant dense<[2.0, 4.0, 6.0, 8.0]> : tensor<4xf32>
  %s = tantu.constant 2.0 : f32

  // CHECK: tantu.div_scalar
  %0 = tantu.div_scalar %a, %s : tensor<4xf32>
  return %0 : tensor<4xf32>
}