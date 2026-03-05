// RUN: %tantu-opt %s | FileCheck %s

func.func @test_transpose() -> tensor<3x2xf32> {
  %a = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>
  // CHECK: tantu.transpose
  %0 = tantu.transpose %a {perm = [1, 0]} : tensor<2x3xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}