#ifndef HELLO_DIALECT
#define HELLO_DIALECT

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TinyGrad dialect definition.
//===----------------------------------------------------------------------===//

def TinyGradDialect : Dialect {
    let name = "tinygrad";
    let summary = "The tinygrad dialect.";
    let description = [{
        This dialect is to represent a target IR for tinygrad nn lib, from which computations
        can be lowered to different dialects, such as gpu, so the accel part of tinygrad will 

    }];
    let cppNamespace = "::tinygrad";
    let hasConstantMaterializer = 1;
}

//===----------------------------------------------------------------------===//
// Base hello operation definition.
//===----------------------------------------------------------------------===//

class TinyGradOp<string mnemonic, list<Trait> traits = []> :
        Op<TinyGradDialect, mnemonic, traits>;

#endif // TINYGRAD_DIALECT
