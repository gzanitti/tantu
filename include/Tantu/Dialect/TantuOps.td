include "Tantu/Dialect/TantuDialect.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.td"
include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/BuiltinAttributes.td"

class Tantu_Op<string mnemonic, list<Trait> traits = []> :
    Op<Tantu_Dialect, mnemonic, traits>;

// Constants

def Tantu_ConstantOp : Tantu_Op<"constant", [
  Pure,
  ConstantLike,
  AllTypesMatch<["value", "result"]>
]> {
  let summary = "Produce a scalar constant value";

  let description = [{
    Produces a scalar constant of either f32 or i32 type. The result type
    is determined by the attribute value.

    Examples:
      %0 = tantu.constant 1.0 : f32
      %1 = tantu.constant 42  : i32
  }];

  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let hasVerifier = 1;
  let assemblyFormat = "attr-dict $value";
}

// Elementwise tensor-tensor operations

def Tantu_AddOp : Tantu_Op<"add", [
  Pure, SameOperandsAndResultType
]> {
  let summary = "Add two tensors element-wise";

  let description = [{
  Adds two tensors element-wise.

  Examples:
    %0 = tantu.add %a, %b : tensor<2x2xf32>
  }];

  let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Tantu_SubOp : Tantu_Op<"sub", [
  Pure, SameOperandsAndResultType
]> {
  let summary = "Subtract two tensors element-wise";

  let description = [{
  Subtracts two tensors element-wise.

  Examples:
    %0 = tantu.sub %a, %b : tensor<2x2xf32>
  }];

  let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Tantu_MulOp : Tantu_Op<"mul", [
  Pure, SameOperandsAndResultType
]> {
  let summary = "Multiplies two tensors element-wise";

  let description = [{
  Multiplies two tensors element-wise.

  Examples:
    %0 = tantu.mul %a, %b : tensor<2x2xf32>
  }];

  let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}


def Tantu_DivOp : Tantu_Op<"div", [
  Pure, SameOperandsAndResultType
]> {
  let summary = "Divides two tensors element-wise";

  let description = [{
  Divides two tensors element-wise.

  Examples:
    %0 = tantu.div %a, %b : tensor<2x2xf32>
  }];

  let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Tantu_MaxOp : Tantu_Op<"max", [
  Pure, SameOperandsAndResultType
]> {
  let summary = "Computes the maximum of two tensors element-wise";

  let description = [{
  Computes the maximum of two tensors element-wise.

  Examples:
    %0 = tantu.max %a, %b : tensor<2x2xf32>
  }];

  let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def Tantu_ExpOp : Tantu_Op<"exp", [
  Pure, SameOperandsAndResultType
]> {
  let summary = "Exponentiates a tensor element-wise";

  let description = [{
  Exponentiates a tensor element-wise.

  Examples:
    %0 = tantu.exp %a : tensor<2x2xf32>
  }];

  let arguments = (ins F32Tensor:$input);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

def Tantu_NegOp : Tantu_Op<"neg", [
  Pure, SameOperandsAndResultType
]> {
  let summary = "Negates a tensor element-wise";

  let description = [{
  Negates a tensor element-wise.

  Examples:
    %0 = tantu.neg %a : tensor<2x2xf32>
  }];

  let arguments = (ins F32Tensor:$input);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

// Tensor-scalar operations

def Tantu_AddScalarOp : Tantu_Op<"add_scalar", [
  Pure,
  AllTypesMatch<["input", "result"]>
]> {
  let summary = "Add a scalar to a tensor element-wise";

  let description = [{
  Adds a scalar to a tensor element-wise.

  Examples:
    %0 = tantu.add_scalar %a, 1.0 : tensor<2x2xf32>
  }];

  let arguments = (ins F32Tensor:$input, F32:$scalar);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$input `,` $scalar attr-dict `:` type($result)";
}

def Tantu_SubScalarOp : Tantu_Op<"sub_scalar", [
  Pure,
  AllTypesMatch<["input", "result"]>
]> {
  let summary = "Subtract a scalar from a tensor element-wise";

  let description = [{
  Subtract a scalar from a tensor element-wise.

  Examples:
    %0 = tantu.sub_scalar %a, 1.0 : tensor<2x2xf32>
  }];

  let arguments = (ins F32Tensor:$input, F32:$scalar);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$input `,` $scalar attr-dict `:` type($result)";
}

def Tantu_MulScalarOp : Tantu_Op<"mul_scalar", [
  Pure,
  AllTypesMatch<["input", "result"]>
]> {
  let summary = "Multiply a scalar to a tensor element-wise";

  let description = [{
  Multiplies a scalar to a tensor element-wise.

  Examples:
    %0 = tantu.mul_scalar %a, 1.0 : tensor<2x2xf32>
  }];

  let arguments = (ins F32Tensor:$input, F32:$scalar);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$input `,` $scalar attr-dict `:` type($result)";
}

def Tantu_DivScalarOp : Tantu_Op<"div_scalar", [
  Pure,
  AllTypesMatch<["input", "result"]>
]> {
  let summary = "Divides a scalar from a tensor element-wise";

  let description = [{
  Divides a scalar from a tensor element-wise.

  Examples:
    %0 = tantu.div_scalar %a, 1.0 : tensor<2x2xf32>
  }];

  let arguments = (ins F32Tensor:$input, F32:$scalar);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$input `,` $scalar attr-dict `:` type($result)";
}

def Tantu_MaxScalarOp : Tantu_Op<"max_scalar", [
  Pure,
  AllTypesMatch<["input", "result"]>
]> {
  let summary = "Computes the maximum of a tensor and a scalar element-wise";

  let description = [{
  Computes the maximum of a tensor and a scalar element-wise.

  Examples:
    %0 = tantu.max_scalar %a, 1.0 : tensor<2x2xf32>
  }];

  let arguments = (ins F32Tensor:$input, F32:$scalar);
  let results = (outs F32Tensor:$result);
  let assemblyFormat = "$input `,` $scalar attr-dict `:` type($result)";
}

// Reduction operations

def Tantu_SumOp : Tantu_Op<"sum", [
  Pure,
  SameOperandsElementType
]> {
  let summary = "Computes the sum of all elements in a tensor";

  let description = [{
  Computes the sum of all elements in a tensor.

  Examples:
    %0 = tantu.sum %a : tensor<2x2xf32> -> f32
  }];

  let arguments = (ins F32Tensor:$input);
  let results = (outs F32:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Tantu_MaxReduceOp : Tantu_Op<"max_reduce", [
  Pure,
  SameOperandsElementType
]> {
  let summary = "Computes the maximum of all elements in a tensor";

  let description = [{
  Computes the maximum of all elements in a tensor.
    
  Examples:
    %0 = tantu.max_reduce %a : tensor<2x2xf32> -> f32
  }];

  let arguments = (ins F32Tensor:$input);
  let results = (outs F32:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}

def Tantu_SizeOp : Tantu_Op<"size", [
  Pure
]> {
  let summary = "Computes the size of a tensor";

  let description = [{
  Computes the size of a tensor.

  Examples:
    %0 = tantu.size %a : tensor<2x2xf32> -> i32
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs I32:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}


// Type convertion operations

def Tantu_CastOp : Tantu_Op<"cast", [
  Pure
]> {
  let summary = "Convert `i32` scalar to `f32`";

  let description = [{
  Convert `i32` scalar to `f32`.

  Examples:
    %0 = tantu.cast %a : i32 -> f32
  }];

  let arguments = (ins I32:$input);
  let results = (outs F32:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

}

// Linear algebra operations

def Tantu_TransposeOp : Tantu_Op<"transpose", [
  Pure
]> {
  let summary = "Transpose a tensor based on the permutation vector";

  let description = [{
  Transposes a tensor based on the permutation vector.
  
  Example:
    %0 = tantu.transpose %a {perm = [1, 0]} : tensor<2x3xf32> -> tensor<3x2xf32>
  }];

  let arguments = (ins F32Tensor:$input, ArrayAttr:$perm);
  let results = (outs F32Tensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";

}

def Tantu_MatmulOp : Tantu_Op<"matmul", [
  Pure
]> {
  let summary = "Matrix multiplication of two tensors";

  let description = [{
  Computes the matrix multiplication of two tensors.

  Examples:
    %0 = tantu.matmul %a, %b : tensor<2x3xf32>, tensor<3x2xf32> -> tensor<2x2xf32>
  }];

  let arguments = (ins F32Tensor:$lhs, F32Tensor:$rhs);
  let results = (outs F32Tensor:$result);
  let hasVerifier = 1;
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";

}

// I/O operations

def Tantu_PrintOp : Tantu_Op<"print", [
  MemoryEffects<[MemWrite<DefaultResource>]>,
  DeclareOpInterfaceMethods<BufferizableOpInterface, [
  "bufferizesToMemoryRead",
  "bufferizesToMemoryWrite", 
  "getAliasingValues",
  "bufferize"
  ]>
]> {
  let summary = "Print a tensor to the console";

  let description = [{
  Prints a tensor to the console.

  Examples:
    tantu.print %a : tensor<2x2xf32>
  }];

  let arguments = (ins AnyTypeOf<[F32Tensor, AnyMemRef]>:$input);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}