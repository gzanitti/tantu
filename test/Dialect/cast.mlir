// RUN: %tantu-opt %s | FileCheck %s

func.func @test_cast_i32_to_f32() -> f32 {
  %a = tantu.constant 42 : i32

  // CHECK: tantu.cast
  %0 = tantu.cast %a : i32 -> f32
  return %0 : f32
}