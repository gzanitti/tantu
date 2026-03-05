// RUN: %tantu-opt %s | FileCheck %s

func.func @test_max_reduce_f32() -> f32 {
  %a = arith.constant dense<[3.0, 1.0, 4.0, 1.0]> : tensor<4xf32>

  // CHECK: tantu.max_reduce
  %0 = tantu.max_reduce %a : tensor<4xf32> -> f32
  return %0 : f32
}