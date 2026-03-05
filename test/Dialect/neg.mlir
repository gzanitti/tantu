// RUN: %tantu-opt %s | FileCheck %s

func.func @test_neg_f32() -> tensor<4xf32> {
  %a = arith.constant dense<[1.0, -2.0, 3.0, -4.0]> : tensor<4xf32>

  // CHECK: tantu.neg
  %0 = tantu.neg %a : tensor<4xf32>
  return %0 : tensor<4xf32>
}