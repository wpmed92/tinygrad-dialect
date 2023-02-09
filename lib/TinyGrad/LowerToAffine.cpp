// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#include "TinyGrad/TinyGradDialect.h"
#include "TinyGrad/TinyGradOps.h"
#include "TinyGrad/TinyGradPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/APFloat.h"
#include <type_traits>
#include <iostream>

static mlir::MemRefType convertTensorToMemRef(mlir::TensorType type) {
  assert(type.hasRank() && "expected only ranked shapes");
  return mlir::MemRefType::get(type.getShape(), type.getElementType());
}

static mlir::Value insertAllocAndDealloc(mlir::MemRefType type, mlir::Location loc,
                                         mlir::PatternRewriter &rewriter) {
  auto alloc = rewriter.create<mlir::memref::AllocOp>(loc, type);

  // Make sure to allocate at the beginning of the block.
  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  // Make sure to deallocate this alloc at the end of the block. This is fine
  // as toy functions have no control flow.
  auto dealloc = rewriter.create<mlir::memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

class ConstantOpLowering : public mlir::OpRewritePattern<tinygrad::ConstantOp> {
  using OpRewritePattern<tinygrad::ConstantOp>::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(tinygrad::ConstantOp op, mlir::PatternRewriter &rewriter) const final {
    mlir::DenseElementsAttr constantValue = op.getValue();
    mlir::Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = op.getType().cast<mlir::TensorType>();
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    mlir::SmallVector<mlir::Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(
          0, *std::max_element(valueShape.begin(), valueShape.end())))
        constantIndices.push_back(rewriter.create<mlir::arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0));
    }
    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.

    // [4, 3] (1, 2, 3, 4, 5, 6, 7, 8)
    // storeElements(0)
    //   indices = [0]
    //   storeElements(1)
    //     indices = [0, 0]
    //     storeElements(2)
    //       store (const 1) [0, 0]
    //     indices = [0]
    //     indices = [0, 1]
    //     storeElements(2)
    //       store (const 2) [0, 1]
    //  ...
    //
    mlir::SmallVector<mlir::Value, 2> indices;
    auto valueIt = constantValue.getValues<mlir::FloatAttr>().begin();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<mlir::AffineStoreOp>(
            loc, rewriter.create<mlir::arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
    return mlir::success();
  }
};

class PrintOpLowering : public mlir::OpConversionPattern<tinygrad::PrintOp> {
  using OpConversionPattern<tinygrad::PrintOp>::OpConversionPattern;

  mlir::LogicalResult matchAndRewrite(tinygrad::PrintOp op, OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const final {
      // We don't lower "hello.print" in this pass, but we need to update its
      // operands.
      rewriter.updateRootInPlace(op,
                                 [&] { op->setOperands(adaptor.getOperands()); });
      return mlir::success();
  }
};

using LoopIterationFn = mlir::function_ref<mlir::Value(
	    mlir::OpBuilder &rewriter, mlir::ValueRange memRefOperands, mlir::ValueRange loopIvs)>;

static void lowerOpToLoops(mlir::Operation *op, mlir::ValueRange operands,
	                           mlir::PatternRewriter &rewriter,
	                           LoopIterationFn processIteration) {
  auto tensorType = (*op->result_type_begin()).cast<mlir::TensorType>();
  auto loc = op->getLoc();

  // Insert an allocation and deallocation for the result of this operation.
  auto memRefType = convertTensorToMemRef(tensorType);
  auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

  // Create a nest of affine loops, with one loop per dimension of the shape.
  // The buildAffineLoopNest function takes a callback that is used to construct
  // the body of the innermost loop given a builder, a location and a range of
  // loop induction variables.
  mlir::SmallVector<int64_t, 4> lowerBounds(tensorType.getRank(), /*Value=*/0);
  mlir::SmallVector<int64_t, 4> steps(tensorType.getRank(), /*Value=*/1);

  buildAffineLoopNest(
      rewriter, loc, lowerBounds, tensorType.getShape(), steps,
      [&](mlir::OpBuilder &nestedBuilder, mlir::Location loc, mlir::ValueRange ivs) {
        // Call the processing function with the rewriter, the memref operands,
        // and the loop induction variables. This function will return the value
        // to store at the current index.
        mlir::Value valueToStore = processIteration(nestedBuilder, operands, ivs);
        nestedBuilder.create<mlir::AffineStoreOp>(loc, valueToStore, alloc, ivs);
      });

  // Replace this operation with the generated alloc.
	  rewriter.replaceOp(op, alloc);
}

// Template specialization helpers for creating lowered binary ops
template <class LoweredBinaryOp>
LoweredBinaryOp buildLoweredBinaryOp(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs)
{
  return builder.create<LoweredBinaryOp>(loc, lhs, rhs);
}

template <>
mlir::arith::CmpFOp buildLoweredBinaryOp(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value lhs, mlir::Value rhs)
{
  return builder.create<mlir::arith::CmpFOp>(loc, mlir::arith::CmpFPredicate::OEQ, lhs, rhs);
}

// Template specialization helpers for creating lowered unary ops
template <class LoweredUnaryOp>
LoweredUnaryOp buildLoweredUnaryOp(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value operand, mlir::MLIRContext *ctx)
{
  return builder.create<LoweredUnaryOp>(loc, operand);
}

template <>
mlir::arith::MaxFOp buildLoweredUnaryOp(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value operand, mlir::MLIRContext *ctx)
{
  double dzero = 0.0f;
  llvm::APFloat apzero = llvm::APFloat(dzero);
  mlir::arith::ConstantFloatOp zero = builder.create<mlir::arith::ConstantFloatOp>(loc, apzero, mlir::FloatType::getF64(ctx));
  return builder.create<mlir::arith::MaxFOp>(loc, operand, zero);
}

template <>
mlir::arith::CmpFOp buildLoweredUnaryOp(mlir::OpBuilder &builder, mlir::Location loc, mlir::Value operand, mlir::MLIRContext *ctx)
{
  double dzero = 0.0f;
  auto apzero = llvm::APFloat(dzero);
  auto zero = builder.create<mlir::arith::ConstantFloatOp>(loc, apzero, mlir::FloatType::getF64(ctx));
  auto result = builder.create<mlir::arith::CmpFOp>(loc,  mlir::arith::CmpFPredicate::OGT, operand, zero);

  return result;
}

template <typename BinaryOp, typename LoweredBinaryOp>struct BinaryOpLowering : public mlir::ConversionPattern {
  BinaryOpLowering(mlir::MLIRContext *ctx)
      : ConversionPattern(BinaryOp::getOperationName(), 1, ctx) {}

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc](mlir::OpBuilder &builder, mlir::ValueRange memRefOperands,
                         mlir::ValueRange loopIvs) {
                     typename BinaryOp::Adaptor binaryAdaptor(memRefOperands);

                     auto loadedLhs = builder.create<mlir::AffineLoadOp>(
                         loc, binaryAdaptor.getLhs(), loopIvs);
                     auto loadedRhs = builder.create<mlir::AffineLoadOp>(
                         loc, binaryAdaptor.getRhs(), loopIvs);

                     return buildLoweredBinaryOp<LoweredBinaryOp>(builder, loc, loadedLhs, loadedRhs);
                   });
    return mlir::success();
  }
};

template <typename UnaryOp, typename LoweredUnaryOp>struct UnaryOpLowering : public mlir::ConversionPattern {
  UnaryOpLowering(mlir::MLIRContext *ctx)
      : ConversionPattern(UnaryOp::getOperationName(), 1, ctx), ctx(ctx) {

      }

  private:
    mlir::MLIRContext *ctx;

  mlir::LogicalResult matchAndRewrite(mlir::Operation *op, mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const final {
    auto loc = op->getLoc();
    lowerOpToLoops(op, operands, rewriter,
                   [loc, this](mlir::OpBuilder &builder, mlir::ValueRange memRefOperands,
                         mlir::ValueRange loopIvs) {
                     typename UnaryOp::Adaptor unaryAdaptor(memRefOperands);

                     auto loadedOperand = builder.create<mlir::AffineLoadOp>(loc, unaryAdaptor.getOperand(), loopIvs);

                     return buildLoweredUnaryOp<LoweredUnaryOp>(builder, loc, loadedOperand, ctx);
                   });
    return mlir::success();
  }
};

// Binary TinyGrad ops
using AddOpLowering = BinaryOpLowering<tinygrad::AddOp, mlir::arith::AddFOp>;
using SubOpLowering = BinaryOpLowering<tinygrad::SubOp, mlir::arith::SubFOp>;
using MulOpLowering = BinaryOpLowering<tinygrad::MulOp, mlir::arith::MulFOp>;
using DivOpLowering = BinaryOpLowering<tinygrad::DivOp, mlir::arith::DivFOp>;
using PowOpLowering = BinaryOpLowering<tinygrad::PowOp, mlir::math::PowFOp>;
using CmpEqLowering = BinaryOpLowering<tinygrad::CmpEq, mlir::arith::CmpFOp>;

// Unary TinyGrad ops
using ReluOpLowering = UnaryOpLowering<tinygrad::ReluOp, mlir::arith::MaxFOp>;
using ExpOpLowering  = UnaryOpLowering<tinygrad::ExpOp,  mlir::math::ExpOp>;
using LogOpLowering  = UnaryOpLowering<tinygrad::LogOp,  mlir::math::LogOp>;
using NegOpLowering  = UnaryOpLowering<tinygrad::NegOp,  mlir::arith::NegFOp>;
using Gt0OpLowering  = UnaryOpLowering<tinygrad::Gt0Op,  mlir::arith::CmpFOp>;

namespace {
class TinyGradToAffineLowerPass : public mlir::PassWrapper<TinyGradToAffineLowerPass, mlir::OperationPass<mlir::ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TinyGradToAffineLowerPass)

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
      registry.insert<mlir::AffineDialect, mlir::arith::ArithDialect, mlir::func::FuncDialect, mlir::memref::MemRefDialect, mlir::math::MathDialect>();
  }

  void runOnOperation() final;
};
}

void TinyGradToAffineLowerPass::runOnOperation() {
  mlir::ConversionTarget target(getContext());

  target.addIllegalDialect<tinygrad::TinyGradDialect>();
  target.addLegalDialect<mlir::AffineDialect, mlir::BuiltinDialect,
    mlir::func::FuncDialect, mlir::arith::ArithDialect, mlir::math::MathDialect, mlir::memref::MemRefDialect>();
  target.addDynamicallyLegalOp<tinygrad::PrintOp>([](tinygrad::PrintOp op) {
      return llvm::none_of(op->getOperandTypes(),
                           [](mlir::Type type) { return type.isa<mlir::TensorType>(); });
  });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<
    ConstantOpLowering, 

    // Binary
    AddOpLowering, 
    SubOpLowering, 
    MulOpLowering, 
    DivOpLowering, 
    PowOpLowering, 
    CmpEqLowering, 

    // Unary
    ReluOpLowering,
    ExpOpLowering,
    LogOpLowering,
    NegOpLowering,
    Gt0OpLowering,

    PrintOpLowering>(&getContext());

  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<mlir::Pass> tinygrad::createLowerToAffinePass() {
  return std::make_unique<TinyGradToAffineLowerPass>();
}