// RUN: %tantu-opt --convert-tantu-to-linalg --one-shot-bufferize="bufferize-function-boundaries=true" %s | FileCheck %s

func.func @test_size_bufferized(%a: tensor<2x4xf32>) -> i32 {
  %0 = tantu.size %a : tensor<2x4xf32> -> i32
  return %0 : i32
}

// CHECK-LABEL: func.func @test_size_bufferized
// CHECK: arith.constant 8 : i32
// CHECK-NOT: tantu.size
