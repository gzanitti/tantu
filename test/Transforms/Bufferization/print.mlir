// RUN: %tantu-opt --convert-tantu-to-linalg --one-shot-bufferize="bufferize-function-boundaries=true" %s | FileCheck %s

func.func @test_print_bufferized(%a: tensor<4xf32>) {
  tantu.print %a : tensor<4xf32>
  return
}

// CHECK-LABEL: func.func @test_print_bufferized
// CHECK: tantu.print
