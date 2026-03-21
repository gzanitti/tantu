#include "Tantu/Dialect/TantuDialect.h"
#include "Tantu/Dialect/TantuOps.h"
#include "Tantu/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Woverloaded-virtual"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#pragma GCC diagnostic pop
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::tantu {

#define GEN_PASS_DEF_TANTUTOLINALGPASS
#include "Passes.h.inc"

struct ConstantOpLowering : public OpConversionPattern<::tantu::ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto constantOp = rewriter.create<arith::ConstantOp>(loc, op.getValue());

    rewriter.replaceOp(op, constantOp.getResult());
    return success();
  }
};

struct TensorLiteralOpLowering
    : public OpConversionPattern<::tantu::TensorLiteralOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::TensorLiteralOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto tensorLiteralOp =
        rewriter.create<arith::ConstantOp>(loc, op.getValue());

    rewriter.replaceOp(op, tensorLiteralOp.getResult());
    return success();
  }
};

// Lowering for elementwise operations

struct AddOpLowering : public OpConversionPattern<::tantu::AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    SmallVector<AffineMap> maps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType,
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, // inputs
        ValueRange{output},                             // outputs
        maps, iterators, [](OpBuilder &b, Location loc, ValueRange args) {
          // args[0] = lhs, args[1] = rhs
          // args[2] = output (not used)
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct SubOpLowering : public OpConversionPattern<::tantu::SubOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    SmallVector<AffineMap> maps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{output}, maps, iterators,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::SubFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct MulOpLowering : public OpConversionPattern<::tantu::MulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    SmallVector<AffineMap> maps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{output}, maps, iterators,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::MulFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct DivOpLowering : public OpConversionPattern<::tantu::DivOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::DivOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    SmallVector<AffineMap> maps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{output}, maps, iterators,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::DivFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct MaxOpLowering : public OpConversionPattern<::tantu::MaxOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::MaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    SmallVector<AffineMap> maps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{output}, maps, iterators,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct NegOpLowering : public OpConversionPattern<::tantu::NegOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::NegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    SmallVector<AffineMap> maps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{adaptor.getInput()}, ValueRange{output},
        maps, iterators, [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::NegFOp>(loc, args[0]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct ExpOpLowering : public OpConversionPattern<::tantu::ExpOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::ExpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    SmallVector<AffineMap> maps(
        2, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{adaptor.getInput()}, ValueRange{output},
        maps, iterators, [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<math::ExpOp>(loc, args[0]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Lowering for tensor-scalar operations

struct AddScalarOpLowering : public OpConversionPattern<::tantu::AddScalarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::AddScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    SmallVector<AffineMap> maps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto tensorType = cast<RankedTensorType>(op.getInput().getType());
    Value splatted =
        rewriter.create<tensor::SplatOp>(loc, adaptor.getScalar(), tensorType);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{adaptor.getInput(), splatted},
        ValueRange{output}, maps, iterators,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct SubScalarOpLowering : public OpConversionPattern<::tantu::SubScalarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::SubScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    SmallVector<AffineMap> maps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto tensorType = cast<RankedTensorType>(op.getInput().getType());
    Value splatted =
        rewriter.create<tensor::SplatOp>(loc, adaptor.getScalar(), tensorType);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{adaptor.getInput(), splatted},
        ValueRange{output}, maps, iterators,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::SubFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct MulScalarOpLowering : public OpConversionPattern<::tantu::MulScalarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::MulScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    SmallVector<AffineMap> maps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto tensorType = cast<RankedTensorType>(op.getInput().getType());
    Value splatted =
        rewriter.create<tensor::SplatOp>(loc, adaptor.getScalar(), tensorType);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{adaptor.getInput(), splatted},
        ValueRange{output}, maps, iterators,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::MulFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct DivScalarOpLowering : public OpConversionPattern<::tantu::DivScalarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::DivScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    SmallVector<AffineMap> maps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto tensorType = cast<RankedTensorType>(op.getInput().getType());
    Value splatted =
        rewriter.create<tensor::SplatOp>(loc, adaptor.getScalar(), tensorType);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{adaptor.getInput(), splatted},
        ValueRange{output}, maps, iterators,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::DivFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

struct MaxScalarOpLowering : public OpConversionPattern<::tantu::MaxScalarOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::MaxScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = cast<RankedTensorType>(op.getType());
    int64_t rank = resultType.getRank();

    SmallVector<AffineMap> maps(
        3, AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext()));

    SmallVector<utils::IteratorType> iterators(rank,
                                               utils::IteratorType::parallel);

    Value output = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto tensorType = cast<RankedTensorType>(op.getInput().getType());
    Value splatted =
        rewriter.create<tensor::SplatOp>(loc, adaptor.getScalar(), tensorType);

    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resultType, ValueRange{adaptor.getInput(), splatted},
        ValueRange{output}, maps, iterators,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    rewriter.replaceOp(op, genericOp.getResult(0));
    return success();
  }
};

// Reduction operations

struct SumOpLowering : public OpConversionPattern<::tantu::SumOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::SumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = op.getType();

    Value emptyOutput =
        rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{}, resultType);

    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    Value output =
        rewriter.create<linalg::FillOp>(loc, zero, emptyOutput).result();

    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    SmallVector<int64_t> dimensions;
    for (int64_t i = 0; i < inputType.getRank(); i++)
      dimensions.push_back(i);

    auto reduceOp = rewriter.create<linalg::ReduceOp>(
        loc, adaptor.getInput(), output, dimensions,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::AddFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    Value result = rewriter.create<tensor::ExtractOp>(
        loc, reduceOp.getResult(0), ValueRange{});

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct MaxReduceOpLowering : public OpConversionPattern<::tantu::MaxReduceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::MaxReduceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = op.getType();

    Value emptyOutput =
        rewriter.create<tensor::EmptyOp>(loc, ArrayRef<int64_t>{}, resultType);

    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    Value output =
        rewriter.create<linalg::FillOp>(loc, zero, emptyOutput).result();

    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    SmallVector<int64_t> dimensions;
    for (int64_t i = 0; i < inputType.getRank(); i++)
      dimensions.push_back(i);

    auto reduceOp = rewriter.create<linalg::ReduceOp>(
        loc, adaptor.getInput(), output, dimensions,
        [](OpBuilder &b, Location loc, ValueRange args) {
          Value sum = b.create<arith::MaximumFOp>(loc, args[0], args[1]);
          b.create<linalg::YieldOp>(loc, sum);
        });

    Value result = rewriter.create<tensor::ExtractOp>(
        loc, reduceOp.getResult(0), ValueRange{});

    rewriter.replaceOp(op, result);
    return success();
  }
};

struct MatMulOpLowering : public OpConversionPattern<::tantu::MatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();

    auto resultType = op.getType();

    Value emptyOutput = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());
    Value zero =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getF32FloatAttr(0.0f));
    Value output =
        rewriter.create<linalg::FillOp>(loc, zero, emptyOutput).result();

    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc, resultType, ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        ValueRange{output});

    rewriter.replaceOp(op, matmulOp.getResult(0));
    return success();
  }
};

struct TransposeOpLowering : public OpConversionPattern<::tantu::TransposeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());

    auto permAttr = op.getPerm();
    SmallVector<int64_t> permValues;
    for (auto attr : permAttr)
      permValues.push_back(llvm::cast<IntegerAttr>(attr).getInt());

    Value emptyOutput = rewriter.create<tensor::EmptyOp>(
        loc, resultType.getShape(), resultType.getElementType());

    auto transposeOp = rewriter.create<linalg::TransposeOp>(
        loc, adaptor.getInput(), emptyOutput, ArrayRef<int64_t>(permValues));

    rewriter.replaceOp(op, transposeOp.getResult());
    return success();
  }
};

struct SizeOpLowering : public OpConversionPattern<::tantu::SizeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::SizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    auto inputType = cast<RankedTensorType>(adaptor.getInput().getType());
    auto shape = inputType.getShape();

    int32_t size = 1;
    for (auto dim : shape)
      size *= dim;

    auto sizeOp = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI32IntegerAttr(size));
    rewriter.replaceOp(op, sizeOp.getResult());
    return success();
  }
};

struct CastOpLowering : public OpConversionPattern<::tantu::CastOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(::tantu::CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    auto input = adaptor.getInput();
    auto targetType = op.getType();

    auto sizeOp = rewriter.create<arith::SIToFPOp>(loc, targetType, input);
    rewriter.replaceOp(op, sizeOp.getResult());

    return success();
  }
};

static void
populateTantuToLinalgConversionPatterns(RewritePatternSet &patterns,
                                        TypeConverter &typeConverter) {

  patterns.add<ConstantOpLowering>(typeConverter, patterns.getContext());

  patterns.add<AddOpLowering>(typeConverter, patterns.getContext());
  patterns.add<SubOpLowering>(typeConverter, patterns.getContext());
  patterns.add<MulOpLowering>(typeConverter, patterns.getContext());
  patterns.add<DivOpLowering>(typeConverter, patterns.getContext());
  patterns.add<MaxOpLowering>(typeConverter, patterns.getContext());
  patterns.add<NegOpLowering>(typeConverter, patterns.getContext());
  patterns.add<ExpOpLowering>(typeConverter, patterns.getContext());

  patterns.add<AddScalarOpLowering>(typeConverter, patterns.getContext());
  patterns.add<SubScalarOpLowering>(typeConverter, patterns.getContext());
  patterns.add<MulScalarOpLowering>(typeConverter, patterns.getContext());
  patterns.add<DivScalarOpLowering>(typeConverter, patterns.getContext());
  patterns.add<MaxScalarOpLowering>(typeConverter, patterns.getContext());

  patterns.add<SumOpLowering>(typeConverter, patterns.getContext());
  patterns.add<MaxReduceOpLowering>(typeConverter, patterns.getContext());
  patterns.add<MatMulOpLowering>(typeConverter, patterns.getContext());
  patterns.add<TransposeOpLowering>(typeConverter, patterns.getContext());

  patterns.add<SizeOpLowering>(typeConverter, patterns.getContext());
  patterns.add<CastOpLowering>(typeConverter, patterns.getContext());
}

struct TantuToLinalgPass
    : public impl::TantuToLinalgPassBase<TantuToLinalgPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ModuleOp module = getOperation();
    TypeConverter typeConverter;
    typeConverter.addConversion([](Type t) { return t; });
    ConversionTarget target(*ctx);
    target.addLegalDialect<linalg::LinalgDialect, arith::ArithDialect,
                           math::MathDialect, func::FuncDialect,
                           tensor::TensorDialect, memref::MemRefDialect>();
    target.addLegalOp<mlir::ModuleOp>();
    target.addLegalOp<::tantu::PrintOp>();
    target.addIllegalDialect<::tantu::TantuDialect>();
    RewritePatternSet patterns(ctx);
    populateTantuToLinalgConversionPatterns(patterns, typeConverter);
    if (failed(applyFullConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace mlir::tantu