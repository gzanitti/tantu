// RUN: %tantu-opt %s | FileCheck %s

func.func @test_sum_f32() -> f32 {
  %a = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>

  // CHECK: tantu.sum
  %0 = tantu.sum %a : tensor<4xf32> -> f32
  return %0 : f32
}