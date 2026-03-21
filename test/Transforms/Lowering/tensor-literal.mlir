// RUN: %tantu-opt --convert-tantu-to-linalg %s | FileCheck %s

func.func @test_tensor_literal() -> tensor<3xf32> {
  %0 = tantu.tensor_literal dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK-LABEL: func.func @test_tensor_literal
// CHECK: arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf32>
// CHECK-NOT: tantu.tensor_literal
