// RUN: %tantu-opt %s | FileCheck %s

func.func @test_div_f32() -> tensor<4xf32> {
  %a = arith.constant dense<[10.0, 9.0, 8.0, 7.0]> : tensor<4xf32>
  %b = arith.constant dense<[2.0, 3.0, 4.0, 7.0]> : tensor<4xf32>

  // CHECK: tantu.div
  %0 = tantu.div %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}