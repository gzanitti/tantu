// RUN: %tantu-opt %s | FileCheck %s

func.func @test_max_f32() -> tensor<4xf32> {
  %a = arith.constant dense<[1.0, 5.0, 3.0, 8.0]> : tensor<4xf32>
  %b = arith.constant dense<[4.0, 2.0, 6.0, 7.0]> : tensor<4xf32>

  // CHECK: tantu.max
  %0 = tantu.max %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}