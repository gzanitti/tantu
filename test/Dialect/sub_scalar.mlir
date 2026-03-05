// RUN: %tantu-opt %s | FileCheck %s

func.func @test_sub_scalar_f32() -> tensor<4xf32> {
  %a = arith.constant dense<[10.0, 20.0, 30.0, 40.0]> : tensor<4xf32>
  %s = tantu.constant 5.0 : f32

  // CHECK: tantu.sub_scalar
  %0 = tantu.sub_scalar %a, %s : tensor<4xf32>
  return %0 : tensor<4xf32>
}