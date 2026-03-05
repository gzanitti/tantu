// RUN: %tantu-opt %s | FileCheck %s

func.func @test_size_i32() -> i32 {
  %a = arith.constant dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf32>

  // CHECK: tantu.size
  %0 = tantu.size %a : tensor<2x3xf32> -> i32
  return %0 : i32
}