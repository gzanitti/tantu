# Tantu Language Specification

> Version 0.3 — work in progress

Tantu is a minimal, functional, array-oriented language for numerical computation.
Functions are pure mappings from tensors to tensors. All tensor shapes are fully
known at compile time and are part of the type. There is no mutation and no
side effects.

---

## 1. Types

### 1.1 Scalar Types

| Type  | Description                    |
|-------|--------------------------------|
| `f32` | 32-bit IEEE 754 floating point |
| `i32` | 32-bit signed integer          |

Scalar literals are implicitly typed: `0.0` is `f32`, `0` is `i32`. There are no
implicit conversions between `f32` and `i32`.

### 1.2 Tensor Types

Tensors are the primary data structure. Their type encodes both the element type and
the full static shape:

```
tensor<D0 , D1 , ... , Dn>
```

Examples:

```
tensor<4,4>      -- 4x4 matrix of f32
tensor<8>        -- vector of 8 f32 elements
tensor<2,3,4>    -- rank-3 tensor of f32
```

Tensors have value semantics — they are immutable. Operations produce new tensors
rather than modifying existing ones.

---

## 2. Program Structure

A Tantu program is a sequence of constant definitions and function definitions.

```
program      ::= (const_def | function_def)*
```

### 2.1 Constants

```
const_def ::= 'const' identifier ':' scalar_type '=' scalar_literal ';'
```

Constants are module-level scalar values. They are immutable and can be referenced
from any function defined after them.

Example:

```
const SCALE: f32 = 0.5;
```

### 2.2 Functions

```
function_def ::= 'fn' identifier '(' param_list ')' '->' type '{' body '}'
param_list   ::= (param (',' param)*)?
param        ::= identifier ':' type
body         ::= let_binding* expression
```

A function body consists of zero or more `let` bindings followed by a single return
expression. The return expression must have the type declared in the function
signature. Functions can call other functions defined in the same program.

### 2.3 Let Bindings

```
let_binding ::= 'let' identifier '=' expression ';'
```

`let` introduces a named value. A name cannot be rebound once introduced within the
same scope. The type of the binding is inferred from the right-hand side expression.

### 2.4 Return Expression

The last expression in a function body (not terminated by `;`) is the return value.
There is no explicit `return` keyword.

---

## 3. Expressions

```
expression ::= call_expression
             | tensor_expression
             | identifier
             | scalar_literal

tensor_expression ::= 'tensor' '<' shape '>' '(' (scalar_literal (',' scalar_literal)*)? ')'
shape             ::= integer_literal (',' integer_literal)*
```

All operations are expressed as function calls. There are no binary operators.

```
call_expression ::= identifier '(' argument_list ')'
argument_list   ::= (expression (',' expression)*)?
scalar_literal  ::= float_literal | integer_literal
float_literal   ::= [0-9]+ '.' [0-9]+
integer_literal ::= [0-9]+
```

---

## 4. Operations

### 4.1 Elementwise Operations

Operate component-wise over tensors of identical shape, or between a tensor and a
scalar (the scalar is applied uniformly to all elements).

| Operation          | Signature                                         | Description                       |
|--------------------|---------------------------------------------------|-----------------------------------|
| `add(a, b)`        | `tensor<S>, tensor<S> -> tensor<S>`  | Elementwise addition              |
| `sub(a, b)`        | `tensor<S>, tensor<S> -> tensor<S>`  | Elementwise subtraction           |
| `mul(a, b)`        | `tensor<S>, tensor<S> -> tensor<S>`  | Elementwise multiplication        |
| `div(a, b)`        | `tensor<S>, tensor<S> -> tensor<S>`  | Elementwise division              |
| `neg(a)`           | `tensor<S> -> tensor<S>`                  | Elementwise negation              |
| `exp(a)`           | `tensor<S> -> tensor<S>`                  | Elementwise exponential           |
| `max(a, b)`        | `tensor<S>, tensor<S> -> tensor<S>`  | Elementwise maximum               |
| `add_scalar(a, s)` | `tensor<S>, f32 -> tensor<S>`            | Add scalar to all elements        |
| `sub_scalar(a, s)` | `tensor<S>, f32 -> tensor<S>`            | Subtract scalar from all elements |
| `mul_scalar(a, s)` | `tensor<S>, f32 -> tensor<S>`            | Multiply all elements by scalar   |
| `div_scalar(a, s)` | `tensor<S>, f32 -> tensor<S>`            | Divide all elements by scalar     |
| `max_scalar(a, s)` | `tensor<S>, f32 -> tensor<S>`            | Elementwise max with scalar       |

Where `S` denotes an arbitrary static shape that must be identical for both operands
in the tensor-tensor variants.

### 4.2 Reduction Operations

Reduce a tensor along all dimensions to a scalar.

| Operation       | Signature              | Description               |
|-----------------|------------------------|---------------------------|
| `sum(a)`        | `tensor<S> -> f32` | Sum of all elements       |
| `max_reduce(a)` | `tensor<S> -> f32` | Maximum element           |
| `size(a)`       | `tensor<S> -> i32`     | Total number of elements  |

`max_reduce` is named explicitly to avoid ambiguity with the elementwise `max`.
`size` returns the total number of elements as `i32` and works on tensors of any
element type.

### 4.3 Linear Algebra Operations

| Operation                | Signature                                                        | Description                     |
|--------------------------|------------------------------------------------------------------|---------------------------------|
| `matmul(a, b)`           | `tensor<M,K>, tensor<K,N> -> tensor<M,N>`           | Matrix multiply                 |
| `transpose(a, perm)`     | `tensor<D0,...,Dn>, [i64] -> tensor<permuted_shape>` | Permute tensor dimensions       |

`perm` is an array of integers specifying the permutation of dimensions. It must be
a valid permutation of `[0, 1, ..., rank-1]` where `rank` is the rank of `a`.
The result shape is the shape of `a` reordered according to `perm` — if `a` has
shape `[D0, D1, ..., Dn]` and `perm` is `[p0, p1, ..., pn]`, the result has
shape `[D_p0, D_p1, ..., D_pn]`.

### 4.4 Type Conversion

| Operation        | Signature               | Description                  |
|------------------|-------------------------|------------------------------|
| `cast(a)`  | `i32 -> f32`            | Convert `i32` scalar to `f32`|

Type conversions are always explicit. There are no implicit coercions between `i32`
and `f32`.

### 4.5 I/O Operations

| Operation   | Signature                  | Description                        |
|-------------|----------------------------|------------------------------------|
| `print(a)`  | `tensor<S> -> ()`      | Print tensor contents to stdout    |

`print` is a side-effecting operation — it is the only operation in Tantu that
produces a side effect. It is intended for debugging and result inspection.
Because it has side effects, it must not be marked `Pure` and must not be
eliminated by dead code elimination passes.

### 4.6 Tensor Construction

| Operation | Signature | Description |
|---|---|---|
| `tensor<S>(e0, e1, ..., en)` | `f32,... -> tensor<S>` | Construct a tensor from scalar literals |

`S` is a static shape (e.g. `3x2`, `4x4x4`). The number of arguments must equal
the total number of elements in the shape. Element type is inferred from the literals
and must be uniform across all arguments.

Example:
    tensor<2,3>(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

---

## 5. Semantic Restrictions

The following are compile-time errors:

- Operands of tensor-tensor elementwise operations must have identical shapes.
- `matmul(a, b)`: the last dimension of `a` must equal the first dimension of `b`.
- `transpose(a, perm)`: `perm` must be a valid permutation of `[0, ..., rank(a)-1]`.
- `transpose(a, perm)`: `perm` must have exactly `rank(a)` elements.
- A `let`-bound name cannot be used before its binding.
- A name cannot be bound more than once in the same function body.
- The type of the return expression must match the declared return type.
- Scalar literals cannot be passed where a tensor is expected, except in the
  scalar variants of elementwise operations.
- `tensor<S>(e0, ..., en)`: the number of arguments must equal the total number of elements in S.
- All arguments to a tensor construction must be scalar literals of uniform type.
---

## 6. Derived Operations

The following common operations are not primitives but can be expressed in Tantu:

**ReLU**
```
fn relu(x: tensor<4>) -> tensor<4> {
    max_scalar(x, 0.0)
}
```

**Softmax**
```
fn softmax(x: tensor<8>) -> tensor<8> {
    let m       = max_reduce(x);
    let shifted = sub_scalar(x, m);
    let exps    = exp(shifted);
    let total   = sum(exps);
    div_scalar(exps, total)
}
```

**Swap batch and sequence dimensions (rank-3 example)**
```
fn swap_batch_seq(x: tensor<4,8,16>) -> tensor<8,4,16> {
    transpose(x, [1, 0, 2])
}
```

---
