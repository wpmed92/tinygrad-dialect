#ifndef HELLO_OPS
#define HELLO_OPS

include "TinyGradDialect.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


/*
Buffer                                                     # class of memory on this device
unary_op  (RELU, EXP, LOG, NEG, GT0)                       # A -> A
reduce_op (SUM, MAX)                                       # A -> B (smaller size, B has 1 in shape)
binary_op (ADD, SUB, MUL, DIV, POW, CMPEQ)                 # A + B -> C (all the same size) -> element-wise
movement_op (RESHAPE, PERMUTE, PAD, SHRINK, EXPAND, FLIP)  # A -> B (different size)
processing_op (CONV)                                       # A + B -> C
*/

def ConstantOp : TinyGradOp<"constant", [NoMemoryEffect]> {
  let summary = "constant";
  let description = [{
    Constant operation turns a literal into an SSA value. The data is attached
    to the operation as an attribute. For example:

    ```mlir
      %0 = "tinygrad.constant"()
      { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
      : () -> tensor<2x3xf64>
    ```
  }];

  let builders = [
    OpBuilder<(ins "mlir::DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    OpBuilder<(ins "double":$value)>
  ];

//  let parser = [{ return ::parseConstantOp(parser, result); }];
  let arguments = (ins F64ElementsAttr:$value);
  let results = (outs F64Tensor);
}

class UnaryOp<string mneomic, list<Trait> traits = []> : TinyGradOp<mneomic, traits> {
  let summary = "Base class for unary tinygrad operations";
  let description = [{
    Performs a unary operation on input tensor, and returns the resulting tensor.

    ```mlir
      %0 = "tinygrad.[unaryop]"(%1) tensor<2x2xf64> -> tensor<2x2xf64>
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[F64Tensor, F64MemRef, I1Tensor, I1MemRef]>:$operand
  );

  let results = (outs 
    AnyTypeOf<[F64Tensor, F64MemRef, I1Tensor, I1MemRef]>
  );
}

class BinaryOp<string mnemonic, list<Trait> traits = []> : TinyGradOp<mnemonic, traits>  {
  let summary = "Base class for binary tinygrad operations";
  let description = [{
    Performs a binary operation on two tensors, and returns the resulting tensor.

    ```mlir
      %0 = "tinygrad.[binaryop]"(%1, %2) tensor<2x2xf64>, tensor<2x2xf64> -> tensor<2x2xf64>
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[F64Tensor, F64MemRef]>:$lhs,
    AnyTypeOf<[F64Tensor, F64MemRef]>:$rhs
  );

  let results = (outs AnyTypeOf<[F64Tensor, F64MemRef, I1Tensor]>);
}

// Binary
def AddOp : BinaryOp<"add",   [NoMemoryEffect]>;
def SubOp : BinaryOp<"sub",   [NoMemoryEffect]>;
def MulOp : BinaryOp<"mul",   [NoMemoryEffect]>;
def DivOp : BinaryOp<"div",   [NoMemoryEffect]>;
def PowOp : BinaryOp<"pow",   [NoMemoryEffect]>;
def CmpEq : BinaryOp<"cmpeq", [NoMemoryEffect]>;

// Unary
def ReluOp : UnaryOp<"relu", [NoMemoryEffect]>;
def ExpOp  : UnaryOp<"exp",  [NoMemoryEffect]>;
def LogOp  : UnaryOp<"log",  [NoMemoryEffect]>;
def NegOp  : UnaryOp<"neg",  [NoMemoryEffect]>;
def Gt0Op  : UnaryOp<"gt0",  [NoMemoryEffect]>;

class MovementOp<string mnemonic, list<Trait> traits = []> : TinyGradOp<mnemonic, traits>   {
  let summary = "Base class for movement tinygrad operations";
  let description = [{
    Performs a movement operation on the tensor operand, and returns the resulting tensor.

    ```mlir
      %0 = "tinygrad.[movementop]"(%1) tensor<3x2xf64> -> tensor<6xf64>
    ```
  }];

  let arguments = (ins
    AnyTypeOf<[F64Tensor, F64MemRef]>:$operand
  );

  let results = (outs 
    AnyTypeOf<[F64Tensor, F64MemRef]>
  );
}

def ReshapeOp : MovementOp<"reshape"> {
  let summary = "Reshape movement operation";
  let description = [{
    Reshapes the input tensor to a shape defined in 'shape' attribute.

    ```mlir
      %0 = "tinygrad.constant"()
      { value = dense<[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]> : tensor<2x3xf64> }
      : () -> tensor<2x3xf64>
      %1 = "tinygrad.reshape"(%0)
      { shape = dense<[3,2]> : tensor<2xi32> }
      : (tensor<2x3xf64>) -> tensor<3x2xf64>
    ```
  }];

  let arguments = (ins
    I32ElementsAttr:$shape,
    AnyTypeOf<[F64Tensor, F64MemRef]>:$operand
  );

  let results = (outs AnyTypeOf<[F64Tensor, F64MemRef]>);
}

def PermuteOp : MovementOp<"permute">;
def PadOp     : MovementOp<"pad">;
def ShrinkOp  : MovementOp<"shrink">;
def ExpandOp  : MovementOp<"expand">;
def FlipOp    : MovementOp<"flip">;

def PrintOp : TinyGradOp<"print", [NoMemoryEffect]> {
    let summary = "print operation";
    let description = [{
        The "print" builtin operation prints a given input tensor, and produces
        no results.
    }];

    // The print operation takes an input tensor to print.
    let arguments = (ins AnyTypeOf<[F64Tensor, F64MemRef, I1Tensor, I1MemRef]>:$input);

    let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // HELLO_OPS
