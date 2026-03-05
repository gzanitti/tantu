// RUN: %tantu-opt %s | FileCheck %s

func.func @test_constant_f32() -> f32 {
  // CHECK: tantu.constant 1.000000e+00 : f32
  %0 = tantu.constant 1.0 : f32
  return %0 : f32
}

func.func @test_constant_i32() -> i32 {
  // CHECK: tantu.constant 42 : i32
  %1 = tantu.constant 42 : i32
  return %1 : i32
}