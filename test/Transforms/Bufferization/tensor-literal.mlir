// RUN: %tantu-opt --convert-tantu-to-linalg --one-shot-bufferize="bufferize-function-boundaries=true" %s | FileCheck %s

func.func @test_tensor_literal_bufferized() -> tensor<3xf32> {
  %0 = tantu.tensor_literal dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  return %0 : tensor<3xf32>
}

// CHECK: memref.global {{.*}} : memref<3xf32> = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]>
// CHECK-LABEL: func.func @test_tensor_literal_bufferized
// CHECK: memref.get_global
// CHECK-NOT: tantu.tensor_literal
