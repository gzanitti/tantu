// RUN: %tantu-opt --convert-tantu-to-linalg --one-shot-bufferize="bufferize-function-boundaries=true" %s | FileCheck %s

func.func @test_constant_bufferized() -> f32 {
  %0 = tantu.constant 1.0 : f32
  return %0 : f32
}

// CHECK-LABEL: func.func @test_constant_bufferized
// CHECK: arith.constant 1.000000e+00 : f32
// CHECK-NOT: tantu.constant
