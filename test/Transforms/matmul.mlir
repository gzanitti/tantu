// RUN: %tantu-opt --convert-tantu-to-linalg %s | FileCheck %s

func.func @test_matmul(%a: tensor<4x4xf32>, %b: tensor<4x4xf32>) -> tensor<4x4xf32> {
  %0 = tantu.matmul %a, %b : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
  return %0 : tensor<4x4xf32>
}

// CHECK-LABEL: func.func @test_matmul
// CHECK: tensor.empty
// CHECK: linalg.fill
// CHECK: linalg.matmul
// CHECK-NOT: tantu.matmul