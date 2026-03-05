// RUN: %tantu-opt --convert-tantu-to-linalg --one-shot-bufferize="bufferize-function-boundaries=true" --lower-tantu-print %s | FileCheck %s

func.func @test_print(%a: tensor<4xf32>) {
  tantu.print %a : tensor<4xf32>
  return
}

// CHECK: func.func private @printMemrefF32(memref<*xf32>)
// CHECK-LABEL: func.func @test_print
// CHECK: memref.cast
// CHECK: call @printMemrefF32
// CHECK-NOT: tantu.print