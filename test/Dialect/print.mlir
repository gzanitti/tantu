// RUN: %tantu-opt %s | FileCheck %s

func.func @test_print() {
  %a = arith.constant dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>

  // CHECK: tantu.print
  tantu.print %a : tensor<2x2xf32>
  return
}