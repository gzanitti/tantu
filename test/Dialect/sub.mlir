// RUN: %tantu-opt %s | FileCheck %s

func.func @test_add_f32() -> tensor<4xf32> {
  %a = arith.constant dense<[1.0, 7.0, 7.0, 4.0]> : tensor<4xf32>
  %b = arith.constant dense<[1.0, 3.0, 7.0, 8.0]> : tensor<4xf32>

  // CHECK: tantu.sub
  %0 = tantu.sub %a, %b : tensor<4xf32>
  return %0 : tensor<4xf32>
}