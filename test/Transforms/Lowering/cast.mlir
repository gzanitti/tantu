// RUN: %tantu-opt --convert-tantu-to-linalg %s | FileCheck %s

func.func @test_cast(%a: i32) -> f32 {
  %0 = tantu.cast %a : i32 -> f32
  return %0 : f32
}

// CHECK-LABEL: func.func @test_cast
// CHECK: arith.sitofp
// CHECK-NOT: tantu.cast