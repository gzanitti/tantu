// RUN: %tantu-opt --convert-tantu-to-linalg --fuse-elementwise %s | FileCheck %s

func.func @test_fuse(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = tantu.add %a, %b : tensor<4x4xf32>
  %1 = tantu.neg %0 : tensor<4x4xf32>
  return %1 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_fuse 
// CHECK-COUNT-1: linalg.generic 
// CHECK: arith.addf 
// CHECK: arith.negf