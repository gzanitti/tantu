// RUN: %tantu-opt --convert-tantu-to-linalg %s | FileCheck %s

func.func @test_print(%a: tensor<4xf32>) {
  tantu.print %a : tensor<4xf32>
  return
}

// CHECK-LABEL: func.func @test_print
// CHECK: tantu.print