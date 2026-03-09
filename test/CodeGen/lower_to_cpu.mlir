// RUN: %tantu-opt %s \
// RUN:   --convert-tantu-to-linalg \
// RUN:   --one-shot-bufferize \
// RUN:   --lower-tantu-print \
// RUN:   --lower-to-cpu | FileCheck %s

// CHECK: llvm.func @main
// CHECK: llvm.cond_br
// CHECK: llvm.fadd
// CHECK: llvm.call @printMemrefF32

func.func @main() {
  %a = arith.constant dense<[1.0, 2.0, 3.0, 4.0]> : tensor<4xf32>
  %b = arith.constant dense<[10.0, 20.0, 30.0, 40.0]> : tensor<4xf32>
  %c = tantu.add %a, %b : tensor<4xf32>
  tantu.print %c : tensor<4xf32>
  return
}