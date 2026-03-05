// RUN: %tantu-opt %s | FileCheck %s

func.func @test_matmul() -> tensor<2x4xf32> {
  %a = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  %b = arith.constant dense<[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]> : tensor<3x4xf32>

  // CHECK: tantu.matmul
  %0 = tantu.matmul %a, %b : tensor<2x3xf32>, tensor<3x4xf32> -> tensor<2x4xf32>
  return %0 : tensor<2x4xf32>
}