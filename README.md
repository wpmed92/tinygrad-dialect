# TinyGrad MLIR Dialect

This is an experimental project with the goal to create an MLIR dialect based on the TinyGrad [llops](https://github.com/geohot/tinygrad#adding-an-accelerator-llops)

## Background

TinyGrad defines only 20 llops. Every neural network you build in TinyGrad will eventually be converted to these llops. With an analogy to RISC, the TinyGrad llops can be seen as a ROSTA (Reduced Operation Set Tensor Architecture). Adding an accelerator to TinyGrad means supporting these operations. We aim to implement the llops as an MLIR dialect, from which different lowerings can be realized. Currently it only supports lowering to LLVM, but a GPU lowering is also planned.

## Prerequisites

* [LLVM](https://llvm.org/)
* [MLIR](https://mlir.llvm.org/)
* [CMake](https://cmake.org/)
* [Ninja](https://ninja-build.org/)

You have to build your own MLIR locally, follow the build instructions [here](https://mlir.llvm.org/getting_started/). 

## Building

1. [Build the LLVM project](https://mlir.llvm.org/getting_started/)

2. Build the `tinygrad-opt` compiler
```sh
mkdir build && cd build
LLVM_DIR=/path/to/llvm-project/build/lib/cmake/llvm \
  MLIR_DIR=/path/to/llvm-project/build/lib/cmake/mlir \
  cmake -G Ninja ..

cmake --build . --target tinygrad-opt
```

## Execution

`tinygrad-opt` will lower the MLIR into LLVM bytecode

```
# Lower MLIR to LLVM IR
$ ./build/bin/tinygrad-opt ./examples/tinygrad.mlir > /path/to/tinygrad.ll

# Execute the code with LLVM interpreter
$ lli /path/to/tinygrad.ll 
```

## Supported TinyGrad llops

### Binary ops
- [x] ADD
- [x] SUB
- [x] MUL
- [x] DIV
- [x] POW
- [x] CMPEQ*

### Unary ops
- [x] RELU
- [x] EXP
- [x] LOG
- [x] NEG
- [x] GT0*

### Movement ops
- [x] RESHAPE
- [ ] PERMUTE
- [ ] PAD
- [ ] SHRINK
- [ ] EXPAND
- [ ] FLIP

### Reduce ops
- [ ] SUM
- [ ] MAX

### Processing ops
- [ ] CONV

*NOTE: The marked operations result in `tensor<?xi1>`, but `tinygrad.print` only supports `tensor<?xf64>` currently. Also, so that the result of these operations can passed to other ops, we have to cast the resulting tensors to `tensor<?xf64>`.

### Examples

Unary op
```mlir
  func.func @main() {
    %0 = "tinygrad.constant"() {value = dense<[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %1 = "tinygrad.neg"(%0) : (tensor<2x3xf64>) -> tensor<2x3xf64>
    "tinygrad.print"(%1) : (tensor<2x3xf64>) -> ()
    return
  }
  Output:
  -2.0 -2.0 -2.0 
  -2.0 -2.0 -2.0
```

Binary op

```mlir
  func.func @main() {
    %0 = "tinygrad.constant"() {value = dense<[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]> : tensor<2x3xf64>} : () -> tensor<2x3xf64>
    %1 = "tinygrad.add"(%0, %0) : (tensor<2x3xf64>, tensor<2x3xf64>) -> tensor<2x3xf64>
    "tinygrad.print"(%1) : (tensor<2x3xf64>) -> ()
    return
  }
  Output:
  4.0 4.0 4.0 
  4.0 4.0 4.0
```

Movement op

```mlir
  func.func @main() {
    %0 = "tinygrad.constant"()  { value = dense<[[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]> : tensor<2x3xf64> } : () -> tensor<2x3xf64>
    %1 = "tinygrad.reshape"(%0) { shape = dense<[3,2]> : tensor<2xi32> } : (tensor<2x3xf64>) -> memref<3x2xf64>
    "tinygrad.print"(%1) : (memref<3x2xf64>) -> ()
    return
  }
  Output:
  2.0 2.0
  2.0 2.0
  2.0 2.0
```
