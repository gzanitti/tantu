// RUN: %tantu-opt --convert-tantu-to-linalg %s | FileCheck %s

func.func @test_transpose(%a: tensor<2x3xf32>) -> tensor<3x2xf32> {
  %0 = tantu.transpose %a {perm = [1, 0]} : tensor<2x3xf32> -> tensor<3x2xf32>
  return %0 : tensor<3x2xf32>
}

// CHECK-LABEL: func.func @test_transpose
// CHECK: tensor.empty
// CHECK: linalg.transpose
// CHECK-NOT: tantu.transpose