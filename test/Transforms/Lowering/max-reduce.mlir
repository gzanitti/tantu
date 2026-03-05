// RUN: %tantu-opt --convert-tantu-to-linalg %s | FileCheck %s

func.func @test_max_reduce(%a: tensor<4x4xf32>) -> f32 {
  %0 = tantu.max_reduce %a : tensor<4x4xf32> -> f32
  return %0 : f32
}

// CHECK-LABEL: func.func @test_max_reduce
// CHECK: tensor.empty
// CHECK: linalg.fill
// CHECK: linalg.reduce
// CHECK: arith.maximumf
// CHECK: tensor.extract
// CHECK-NOT: tantu.max_reduce