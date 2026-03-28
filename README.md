# Tantu

Tantu is a minimal, functional, array-oriented language for numerical computation, built on top of
[MLIR](https://mlir.llvm.org/) 18. It is a **learning project** with no intention of being a production ready language.

The project is intentionally scoped to be buildable by one person while still exercising the same
architectural patterns used in real ML infrastructure work: ODS-based dialect definition,
multi-level lowering, bufferization, affine loop optimization, operator fusion via PDL, and AOT
code generation targeting both CPU and GPU (NVVM/PTX/cubin).


# Language Overview
 
Tantu programs are sequences of pure functions over statically-shaped tensors. There is no
mutation, no implicit broadcasting, and no dynamic dispatch. All tensor shapes are fully known at
compile time and are encoded in the type system. All tensors hold `f32` values — the element type
is fixed and not written in function signatures.
 
```
const SCALE: f32 = 0.5;
 
fn softmax(x: tensor<8>) -> tensor<8> {
    let m       = max_reduce(x);
    let shifted = sub_scalar(x, m);
    let exps    = exp(shifted);
    let total   = sum(exps);
    div_scalar(exps, total)
}
 
fn relu(x: tensor<4>) -> tensor<4> {
    max_scalar(x, 0.0)
}
```


## Compiler Architecture
 
Tantu uses a multi-level lowering pipeline, following the MLIR design philosophy of progressive
lowering through well-defined abstraction levels. The CPU and GPU paths share a common frontend
and mid-level IR, diverging at the `linalg` level.
 
```
Tantu source (.tantu)
        │
        ▼
  Lexer → Parser → AST
        │
        ▼
  TypeChecker (type interning via TypeContext)
        │
        ▼
  IRGen (AST → Tantu MLIR dialect)
        │
        ▼
  TantuToLinalg pass
  [--fuse-elementwise (PDL-based fusion)]
        │
        ▼
  linalg + arith + tensor dialects
  ← bifurcation point: CPU vs GPU →
        │                                    │
        │ --lower-to-cpu                     │ --lower-to-gpu / --lower-to-gpu-opt
        ▼                                    ▼
  Bufferization (one-shot)            linalg tiling (MLIR API / custom)
  memref dialect                      gpu dialect
  LowerTantuPrint pass                NVVM dialect
  linalg → affine → scf → cf         PTX
  LLVM dialect                        cubin
        │                                    │
        ▼                                    ▼
  mlir-translate                      tantu-gpu-runner
  llc                                 (CUDA driver API:
  clang                                load cubin,
  native executable                    alloc device memory,
                                       H2D copy,
                                       launch kernel,
                                       D2H copy + print)

```

 
## Current State
 
| Component | Status |
|---|---|
| Language specification | ✅ Complete (v0.3) |
| Custom Tantu MLIR dialect (ODS) | ✅ Complete |
| TantuToLinalg lowering pass | ✅ Complete |
| Bufferization | ✅ Complete |
| `tantu.print` lowering | ✅ Complete |
| Elementwise op fusion (PDL) | ✅ Complete |
| Frontend: Lexer, Parser, AST, PrettyPrinter | ✅ Complete |
| TypeChecker | ✅ Complete |
| IRGen | ✅ Complete |
| `tantu-compiler` executable | ✅ Complete |
| `--lower-to-cpu` pipeline + AOT execution | ✅ Validated end-to-end |
| FileCheck test suite | ✅ Passing |
| `--lower-to-gpu` pipeline (produces `.cubin`) | ✅ Complete |
| **`tantu-gpu-runner`** | 🔧 In progress |
| `--lower-to-gpu-opt` (MLIR API tiling) | 📋 Planned |
| Custom tiling pass | 📋 Planned |
| Benchmarking: CPU vs GPU vs GPU-opt | 📋 Planned |
 
 
## Dependencies
 
- MLIR / LLVM 18 (tested with 18.1.3, installed via apt at `/usr/lib/llvm-18`)
- CMake (with presets)
- clang (for AOT linking with `-no-pie`)
- CUDA toolkit (for GPU compilation and `tantu-gpu-runner`)
 