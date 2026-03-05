// RUN: %tantu-opt %s | FileCheck %s

func.func @test_exp_f32() -> tensor<4xf32> {
  %a = arith.constant dense<[0.0, 1.0, -1.0, 2.0]> : tensor<4xf32>

  // CHECK: tantu.exp
  %0 = tantu.exp %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}