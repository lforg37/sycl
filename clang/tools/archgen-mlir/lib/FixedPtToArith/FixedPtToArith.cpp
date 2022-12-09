//===- FixedPtToArith.cpp - conversion from FixedPt to Arith --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the conversion of The FixedPt dialect to the Arith
// dialect
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

#include "archgen/FixedPt/PassDetail.h"
#include "archgen/FixedPt/Passes.h"
#include "llvm/Support/raw_ostream.h"

using namespace archgen;
using namespace archgen::fixedpt;
namespace arith = mlir::arith;
namespace func = mlir::func;
namespace memref = mlir::memref;
namespace LLVM = mlir::LLVM;
namespace tensor = mlir::tensor;

namespace {

//===----------------------------------------------------------------------===//
// Building blocks of the conversion
//===----------------------------------------------------------------------===//

/// Describe how FixedPt should be converted to an arith type.
struct FixedPtToArithTypeConverter : public mlir::TypeConverter {
  mlir::MLIRContext *ctx;

  /// Describe conversion for function signatures that are not handled like
  /// other types. because there might be constraints on what can be passed
  /// through function arguments so it is possible to map 1 source type to n
  /// destination types
  ///
  /// But in our case everything is simple simple 1 to 1 mapping from FixedPt
  /// types to Arith types
  ///
  /// The return value is used by FuncOpRewriting to know what it needs to
  /// rewrite the function to. TypeConverter::SignatureConversion is used by the
  /// dialect conversion framework to rewrite block operands
  mlir::FunctionType
  convertSignature(mlir::FunctionType ty,
                   TypeConverter::SignatureConversion &result) {
    /// Converted types of every arguments
    llvm::SmallVector<mlir::Type> argsTy;
    /// Converted types of every return value
    llvm::SmallVector<mlir::Type> retTy;
    argsTy.reserve(ty.getNumInputs());
    retTy.reserve(ty.getNumResults());

    int idx = 0;
    for (auto in : ty.getInputs()) {
      /// Simple 1 to 1 mapping from each type just convert them and add them to
      /// the signature description and the new FunctionType

      mlir::Type newInTy = convertType(in);
      argsTy.push_back(newInTy);
      result.addInputs(idx++, {newInTy});
    }
    for (auto out : ty.getResults()) {
      mlir::Type newOutTy = convertType(out);
      retTy.push_back(newOutTy);
    }

    return mlir::FunctionType::get(ctx, argsTy, retTy);
  }

  /// A FixedPtType is converted to its Arith equivalent
  /// for example:
  ///   !fixedpt.fixedPt<8, -7, "signed"> becomes i16
  ///   !fixedpt.fixedPt<3, -14, "unsigned"> becomes i18
  /// everything is stored in signless because it is easier. to implement this
  /// way, but obviously operations will have the correct signed or unsigned
  /// variant
  mlir::IntegerType convertFixedPt(FixedPtType ty) {
    return mlir::IntegerType::get(ctx, ty.getWidth(),
                                  mlir::IntegerType::Signless);
  }

  FixedPtToArithTypeConverter(mlir::MLIRContext *ctx) : ctx(ctx) {

    /// Add Conversions to the converter
    addConversion([&](FixedPtType ty) { return convertFixedPt(ty); });

    /// Since we generate mlir::IntegerType we need to mlir::IntegerType to be
    /// legal. so it must return it self thought convertType
    addConversion([](mlir::IntegerType ty) { return ty; });

    addConversion([&](mlir::MemRefType ty) {
      return mlir::MemRefType::get(ty.getShape(),
                                   convertType(ty.getElementType()));
    });
    addConversion([&](LLVM::LLVMPointerType ty) {
      return LLVM::LLVMPointerType::get(convertType(ty.getElementType()));
    });
  }
};

/// Pass class implementation
struct ConvertFixedPtToArithPass
    : ConvertFixedPtToArithPassBase<ConvertFixedPtToArithPass> {
  virtual void runOnOperation() override final;
};

/// Description of which operations are legal in input and output.
///
/// FixedPtToArith is a partial conversion, so all operation not explicitly
/// marked illegal will be left untouched. Also all operation that we generate
/// as replacement for FixedPt Ops must be explicitly legal
struct FixedPtToArithTarget : public mlir::ConversionTarget {

  /// Check if a type is legal
  static bool isLegalTypeImpl(mlir::Type ty) {
    /// Recursive case: a FunctionType is legal if all of its composing type are
    /// legal
    if (auto funcTy = ty.dyn_cast<mlir::FunctionType>())
      return isLegalType(funcTy.getInputs()) &&
             isLegalType(funcTy.getResults());
    
    if (auto memrefTy = ty.dyn_cast<mlir::MemRefType>())
      return isLegalType(memrefTy.getElementType());

    if (auto ptrTy = ty.dyn_cast<LLVM::LLVMPointerType>())
      return isLegalType(ptrTy.getElementType());

    /// Leaf case: a leaf type is legal if it is not a FixedPtType
    return !ty.isa<FixedPtType>();
  }

  /// Wrapper to easily operator on TypeRange
  static bool isLegalType(mlir::TypeRange tys) {
    return llvm::all_of(tys, isLegalTypeImpl);
  }

  FixedPtToArithTarget(mlir::MLIRContext &Ctx) : ConversionTarget(Ctx) {

    /// It needs to be converted so it must be illegal
    addIllegalDialect<FixedPtDialect>();

    /// Conversions will emit operations form ArithDialect for Operations
    /// from the FixedPtDialect
    addLegalDialect<arith::ArithDialect>();

    addLegalDialect<tensor::TensorDialect>();

    /// func::FuncOp are legal if there type is legal.
    /// We rewrite func::FuncOp only when at least one of there arguments or
    /// return is a FixedPtType
    addDynamicallyLegalOp<func::FuncOp>(
        [&](func::FuncOp func) { return isLegalType(func.getFunctionType()); });

    /// func::ReturnOp are legal if there type is legal.
    /// We rewrite func::ReturnOp only when at least one of there arguments is
    /// a FixedPtType
    addDynamicallyLegalOp<func::ReturnOp, memref::LoadOp, memref::StoreOp,
                          LLVM::LoadOp, LLVM::StoreOp>(
        [&](mlir::Operation *op) {
          return isLegalType(op->getOperands().getTypes()) &&
                 isLegalType(op->getResultTypes());
        });
  }
};

/// This is a utility class to build conversions from fixed point conversions
/// With arith types and operations
/// It also exposed some utilities convert between arith types
///
/// Note: all mlir::Value here are mlir::IntegerType
class ConversionBuilder {
  /// It will be an instance of our type converter defined above
  mlir::TypeConverter &typeConverter;

  /// Used to perform any change to the IR
  mlir::ConversionPatternRewriter &rewriter;

  /// Location of the original FixedPt op we are replacing
  mlir::Location loc;

  fixedpt::RoundingMode rounding;

public:
  ConversionBuilder(
      mlir::TypeConverter &typeConverter,
      mlir::ConversionPatternRewriter &rewriter, mlir::Location loc,
      fixedpt::RoundingMode rounding = fixedpt::RoundingMode::truncate)
      : typeConverter(typeConverter), rewriter(rewriter), loc(loc),
        rounding(rounding) {}

  /// Extend the type of v to dstTy if needed
  /// Since every mlir::IntegerType is signless the sign is given by isSigned
  mlir::Value maybeExtend(mlir::Value v, mlir::IntegerType dstTy,
                          bool isSigned) {

    /// If the destination is larger then the source we need to add an extend op
    if (v.getType().cast<mlir::IntegerType>().getWidth() < dstTy.getWidth()) {
      /// When an extention is signed or unsigned based on isSigned
      if (isSigned)
        v = rewriter
                .create<arith::ExtSIOp>(
                    loc, rewriter.getIntegerType(dstTy.getWidth()), v)
                .getOut();
      else
        v = rewriter
                .create<arith::ExtUIOp>(
                    loc, rewriter.getIntegerType(dstTy.getWidth()), v)
                .getOut();
    }
    return v;
  }

  /// Truncate the type of v to dstTy if needed
  /// truncation doesn't care about the sign. If the value fits in the result
  /// then the value in the output is the same. Otherwise it will be the N
  /// bottom bits.
  mlir::Value maybeTruncate(mlir::Value v, mlir::IntegerType dstTy) {
    if (v.getType().cast<mlir::IntegerType>().getWidth() > dstTy.getWidth())
      v = rewriter.create<arith::TruncIOp>(loc, dstTy, v).getOut();
    return v;
  }

  /// Shift v left or right depending on the requested shift
  mlir::Value relativeLeftShift(mlir::Value v, int shift, bool isSigned) {
    /// When the shift is 0 v is already we was asked for
    if (shift == 0)
      return v;

    /// Otherwise a left or right shift is need of std::abs(shift)
    /// So we create a constant with that value
    mlir::Value constShift =
        rewriter
            .create<arith::ConstantIntOp>(
                loc, std::abs(shift),
                v.getType().cast<mlir::IntegerType>().getWidth())
            .getResult();

    /// If a left shift is needed create one, sign doesn't matter
    if (shift > 0)
      return rewriter.create<arith::ShLIOp>(loc, v, constShift).getResult();

    /// Otherwise a right shift is needed. Do it signed or unsigned based on
    /// what is needed
    if (isSigned)
      return rewriter.create<arith::ShRSIOp>(loc, v, constShift).getResult();
    return rewriter.create<arith::ShRUIOp>(loc, v, constShift).getResult();
  }

  mlir::Value applyRounding(mlir::Value v, int bitsToBeRemoved, bool isSigned) {
    /// Nothing to round If we are adding bits or we round to zero
    if (bitsToBeRemoved <= 0 || rounding == fixedpt::RoundingMode::truncate)
      return v;

    mlir::IntegerType ty = v.getType().cast<mlir::IntegerType>();

    /// we generate v = v + mask
    /// mask=(1 << (bitsToBeRemoved - 1)) to round equidistant values up
    /// mask=(1 << (bitsToBeRemoved - 1)) - 1 to round equidistant values down
    llvm::APInt mask(ty.getWidth(), 1);
    mask = mask.shl(bitsToBeRemoved - 1);

    /// nearest will be emitted the same as nearest_even_to_up
    if (rounding == fixedpt::RoundingMode::nearest_even_to_down)
      mask = mask - 1;

    mlir::Value maskConstant =
        rewriter
            .create<arith::ConstantOp>(loc, rewriter.getIntegerAttr(ty, mask))
            .getResult();
    mlir::Value add =
        rewriter.create<arith::AddIOp>(loc, v, maskConstant).getResult();
    return add;
  }

  /// The top-level function to emit conversions in arith from one fixed point
  /// format to an other. This function will keep the represented fixed point
  /// value the same(if possible) between the input and output
  mlir::Value buildConversion(mlir::Value v, FixedPtType fixedSrc,
                              FixedPtType fixedDst) {
    assert(typeConverter.convertType(fixedSrc) == v.getType());

    /// The type already match, nothing to do
    if (fixedSrc == fixedDst)
      return v;

    /// Shifts larger than the bitwidth are considered UB by llvm semantics.
    /// So we must select a type such that not shift can be larger then its
    /// bitwidth
    mlir::IntegerType tmpTy =
        typeConverter.convertType(fixedDst.getCommonAddType(fixedSrc))
            .cast<mlir::IntegerType>();
    /// Figure out the arith type of the output
    mlir::IntegerType dstTy =
        typeConverter.convertType(fixedDst).cast<mlir::IntegerType>();

    /// Extend if the output bitwidth is larger then the input bitwidth
    v = maybeExtend(v, tmpTy, fixedSrc.isSigned());

    /// If the lsb was moved we need a shift
    if (fixedSrc.getLsb() != fixedDst.getLsb()) {
      v = applyRounding(v, fixedDst.getLsb() - fixedSrc.getLsb(),
                        fixedSrc.isSigned());
      v = relativeLeftShift(v, fixedSrc.getLsb() - fixedDst.getLsb(),
                            fixedSrc.isSigned());
    }

    /// Truncate if the output bitwidth is smaller then the input bitwidth
    v = maybeTruncate(v, dstTy);

    assert(v.getType() == dstTy);
    return v;
  }

  /// Simply adjust the bitwidth of v such that it becomes dstTy
  mlir::Value truncOrExtend(mlir::Value v, mlir::IntegerType dstTy,
                            bool isSigned) {
    v = maybeExtend(v, dstTy, isSigned);
    v = maybeTruncate(v, dstTy);
    assert(v.getType() == dstTy);
    return v;
  }
};

//===----------------------------------------------------------------------===//
// Descriptions of Conversion Patterns
//===----------------------------------------------------------------------===//
// For arith operations, types of lhs, rhs and result must match
// So most operator the pattern is:
// fixedpt.op(a, b) -> arith.op(convert(a), convert(b))
// convert being a conversion to the output type
//
// For each pattern:
//  - op is the original Operation
//  - adaptor is a mock operation containing the inputs the new operation should
//    have
//  - rewriter is used to performs the edit on the IR
//===----------------------------------------------------------------------===//

/// Pattern for fixedpt::AddOp to arith Ops
struct AddOpLowering : public mlir::OpConversionPattern<AddOp> {
  using base = OpConversionPattern<AddOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(AddOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc(),
                                op.getRounding());
    /// For example it converts:
    /// fixedpt.add(%a, %b) : (fixed<1,-1,u>, fixed<4,-2,s>) -> fixed<4,-1,s>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    /// %tmp1 = arith.extui(%a) : (i3) -> i6
    /// %tmp2 = arith.constant() {value = 1 : i7} : () -> i7
    /// %tmp3 = arith.shrsi(%b, %tmp2) : (i7, i7) -> i7
    /// %tmp4 = arith.trunci(%tmp3) : (i7) -> i6
    /// arith.addi(%tmp1, %tmp4) : (i6, i6) -> i6


    // TODO accumulation is performed here, can be replaced by tree reduction or bitheap

    mlir::SmallVector<mlir::Value> converted;
    auto resType = op.getResult().getType().cast<FixedPtType>();
    auto adapt_args = adaptor.getArgs();
    auto args = op.getArgs();
    for (size_t i = 0 ; i < args.size() ; ++i) {
      converted.push_back(
        converter.buildConversion(adapt_args[i], args[i].getType().cast<FixedPtType>(), resType)
      );
    }
    mlir::Value res = converted[0];
    for (size_t i = 1 ; i < converted.size() ; ++i) {
      res = rewriter.create<arith::AddIOp>(op->getLoc(), res, converted[i])->getResult(0);
    }

    rewriter.replaceOp(op, {res});
    return mlir::success();
  }
};

/// Pattern for fixedpt::SubOp to arith Ops
struct SubOpLowering : public mlir::OpConversionPattern<SubOp> {
  using base = OpConversionPattern<SubOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(SubOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// For example it converts:
    /// fixedpt.sub(%a, %b) : (fixed<4,-2,s>, fixed<3,-9,s>) -> fixed<7,-3,s>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    /// %tmp1 = arith.extsi(%a) : (i7) -> i11
    /// %tmp2 = arith.constant() {value = 1 : i11} : () -> i11
    /// %tmp3 = arith.shli(%tmp1, %tmp2) : (i11, i11) -> i11
    /// %tmp4 = arith.constant() {value = 6 : i13} : () -> i13
    /// %tmp5 = arith.shrsi(%b, %tmp4) : (i13, i13) -> i13
    /// %tmp6 = arith.trunci(%tmp5) : (i13) -> i11
    /// arith.subi(%tmp3, %tmp6) : (i11, i11) -> i11

    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc(),
                                op.getRounding());
    mlir::Value lhs = converter.buildConversion(
        adaptor.getLhs(), op.getLhs().getType().cast<FixedPtType>(),
        op.getResult().getType().cast<FixedPtType>());
    mlir::Value rhs = converter.buildConversion(
        adaptor.getRhs(), op.getRhs().getType().cast<FixedPtType>(),
        op.getResult().getType().cast<FixedPtType>());
    rewriter.replaceOpWithNewOp<arith::SubIOp>(op, lhs, rhs);
    return mlir::success();
  }
};

/// Pattern for fixedpt::ConstantOp to arith::ConstantOp
struct ConstantOpLowering : public mlir::OpConversionPattern<ConstantOp> {
  using base = OpConversionPattern<ConstantOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(ConstantOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// For example converts:
    ///   fixedpt.constant(){valueAttr = #fixed_point<3,1.5>} : !fixedPt<1,-1,u>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    ///   arith.constant() {value = 3 : i3} : () -> i3

    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc());
    mlir::Value v =
        rewriter
            .create<arith::ConstantOp>(
                op->getLoc(),
                mlir::IntegerAttr::get(
                    getTypeConverter()->convertType(op.getResult().getType()),
                    op.getValueAttr().getValue().getValue()))
            .getResult();
    rewriter.replaceOp(op, v);
    return mlir::success();
  }
};

/// Pattern for fixedpt::BitcastOp to nothing. the operand after type rewriting
/// should always be the same type as the expected output
struct BitcastOpLowering : public mlir::OpConversionPattern<BitcastOp> {
  using base = OpConversionPattern<BitcastOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(BitcastOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// The type should already be correct in the adaptor
    assert(adaptor.getInput().getType() ==
           typeConverter->convertType(op.getResult().getType()));
    rewriter.replaceOp(op, adaptor.getOperands());
    return mlir::success();
  }
};

/// Pattern for fixedpt::MulOp to arith Ops
struct MulOpLowering : public mlir::OpConversionPattern<MulOp> {
  using base = OpConversionPattern<MulOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(MulOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// FIXME: The mul is performed on a bitwidth that is much larger then it
    /// could be

    /// For example converts:
    ///   fixedpt.mul(%a, %b):(!fixed<4,-2,s>,!fixed<3,-9,s>) -> !fixed<7,-5,s>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    ///   %tmp1 = arith.extsi(%4) : (i7) -> i19
    ///   %tmp2 = arith.extsi(%18) : (i13) -> i19
    ///   %tmp3 = arith.muli(%tmp1, %tmp2) : (i19, i19) -> i19
    ///   %tmp4 = arith.constant() {value = 6 : i19} : () -> i19
    ///   %tmp5 = arith.shrsi(%tmp3, %tmp4) : (i19, i19) -> i19
    ///   arith.trunci(%tmp5) : (i19) -> i13

    FixedPtType internalTy;
    for (mlir::Type ty : op->getOperandTypes())
      internalTy = internalTy
                       ? internalTy.getCommonMulType(ty.cast<FixedPtType>())
                       : ty.cast<FixedPtType>();
    /// And its arith lowering
    mlir::IntegerType internalIntTy =
        typeConverter->convertType(internalTy).cast<mlir::IntegerType>();

    /// reinterpret extend the width of operands to fit the internalIntTy
    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc(),
                                op.getRounding());
    llvm::SmallVector<mlir::Value> convertedArgs;
    for (auto [a, v] : llvm::zip(adaptor.getOperands(), op->getOperands()))
      convertedArgs.push_back(converter.maybeExtend(
          a, internalIntTy, v.getType().cast<FixedPtType>().isSigned()));

    /// Multiply with it
    mlir::Value res = convertedArgs[0];
    for (size_t i = 1; i < convertedArgs.size(); ++i)
      res = rewriter.create<arith::MulIOp>(op->getLoc(), res, convertedArgs[i])
                ->getResult(0);

    /// Convert result to the requested size
    rewriter.replaceOp(
        op, converter.buildConversion(
                res, internalTy, op.getResult().getType().cast<FixedPtType>()));
    return mlir::success();
  }
};

/// Pattern for fixedpt::DivOp to arith Ops
struct DivOpLowering : public mlir::OpConversionPattern<DivOp> {
  using base = OpConversionPattern<DivOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(DivOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// For example converts:
    ///   fixedpt.div(%a, %b):(!fixed<7,-5,s>, !fixed<7,-3,s>) -> !fixed<7,-9,s>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    ///    %tmp1 = arith.extsi(%a) : (i13) -> i20
    ///    %tmp2 = arith.constant() {value = 7 : i20} : () -> i20
    ///    %tmp3 = arith.shli(%tmp1, %tmp2) : (i20, i20) -> i20
    ///    %tmp4 = arith.extsi(%b) : (i11) -> i20
    ///    %tmp4 = arith.divsi(%tmp3, %tmp4) : (i20, i20) -> i20
    ///    arith.trunci(%tmp4) : (i20) -> i17

    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc(),
                                op.getRounding());
    FixedPtType lhsTy = op.getLhs().getType().cast<FixedPtType>();
    FixedPtType rhsTy = op.getRhs().getType().cast<FixedPtType>();
    FixedPtType outTy = op.getResult().getType().cast<FixedPtType>();
    bool isSigned = lhsTy.isSigned() || rhsTy.isSigned();
    mlir::Value lhs;
    mlir::Value rhs;

    /// precision If we divide without shifting inputs
    int currentPrec = lhsTy.getLsb() - rhsTy.getLsb();

    /// how much lsb do we need to add to lhs to have an output with the right
    /// precision
    int lsbOffset = outTy.getLsb() - currentPrec;

    /// If we need to increase precision
    if (lsbOffset <= 0) {
      /// width of the division operation
      int divWidth = std::max(lhsTy.getWidth() - lsbOffset, rhsTy.getWidth());
      int newLsb = lhsTy.getLsb() + lsbOffset;

      /// Convert lhs and rhs to their new formats (or not)
      lhs = converter.buildConversion(
          adaptor.getLhs(), lhsTy,
          FixedPtType::get(rewriter.getContext(), divWidth + newLsb - 1, newLsb,
                           lhsTy.isSigned()));
      rhs = converter.maybeExtend(
          adaptor.getRhs(), rewriter.getIntegerType(divWidth), rhsTy.isSigned());
    } else {
      /// In this case we need to artificially reduce the precision of the
      /// division to fit in the output without calculating useless bits
      op->dump();
      llvm_unreachable("unimplemented");
    }

    mlir::Value divResult;
    /// Emit the division
    if (isSigned)
      divResult =
          rewriter.create<arith::DivSIOp>(op.getLoc(), lhs, rhs).getResult();
    else
      divResult =
          rewriter.create<arith::DivUIOp>(op.getLoc(), lhs, rhs).getResult();

    /// Remove the now useless high bits
    rewriter.replaceOp(
        op, converter.maybeTruncate(
                divResult,
                typeConverter->convertType(outTy).cast<mlir::IntegerType>()));
    return mlir::success();
  }
};

// Pattern for fixedpt::SelectOp to arith Ops
struct SelectOpLowering : public mlir::OpConversionPattern<SelectOp> {
  using base = OpConversionPattern<SelectOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(SelectOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto key = rewriter.create<arith::IndexCastUIOp>(op->getLoc(), rewriter.getIndexType(), adaptor.getKey());
    auto tensor = rewriter.create<mlir::tensor::FromElementsOp>(key->getLoc(), adaptor.getValues());
    mlir::tensor::ExtractOp extract = rewriter.create<mlir::tensor::ExtractOp>(tensor->getLoc(), tensor, key.getOut());
    rewriter.replaceOp(op, extract.getResult());
    return mlir::success();
  }
};

/// Pattern for fixedpt::ConvertOp to arith Ops
struct ConvertOpLowering : public mlir::OpConversionPattern<ConvertOp> {
  using base = OpConversionPattern<ConvertOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(ConvertOp op, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    /// For example converts:
    ///   fixedpt.convert(%a) : (!fixedPt<7, -9, u>) -> !fixedPt<3, -9, s>
    /// Note: MLIR is abbreviated to fit on one line
    /// To:
    ///   arith.trunci(%a) : (i17) -> i13

    ConversionBuilder converter(*typeConverter, rewriter, op->getLoc(),
                                op.getRounding());
    rewriter.replaceOp(op, converter.buildConversion(
                               adaptor.getInput(),
                               op.getInput().getType().cast<FixedPtType>(),
                               op.getResult().getType().cast<FixedPtType>()));
    return mlir::success();
  }
};

/// Pattern to rewrite the type and blocks of a func::FuncOp
struct FuncOpRewriting : public mlir::OpConversionPattern<func::FuncOp> {
  using base = OpConversionPattern<func::FuncOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(func::FuncOp oldFunc, base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    /// Figure out the new type and arguments rewrite rules
    mlir::TypeConverter::SignatureConversion result(
        oldFunc.getFunctionType().getNumInputs());
    mlir::FunctionType newFuncTy =
        getTypeConverter<FixedPtToArithTypeConverter>()->convertSignature(
            oldFunc.getFunctionType(), result);

    /// Create the new function with the new type and same old symbol name
    auto newFunc = rewriter.create<func::FuncOp>(
        oldFunc.getLoc(), oldFunc.getSymName(), newFuncTy);
    /// Move regions from the old function operations to the new
    rewriter.inlineRegionBefore(oldFunc.getRegion(), newFunc.getRegion(),
                                newFunc.getRegion().end());

    /// Rewrite regions block arguments to have new types
    if (mlir::failed(rewriter.convertRegionTypes(&newFunc.getBody(),
                                                 *typeConverter, &result)))
      return mlir::failure();

    /// Replace uses of oldFunc and notify rewriter we are done
    rewriter.replaceOp(oldFunc, newFunc->getResults());

    return mlir::success();
  }
};

template <typename SimpleOp>
struct SimpleOpRewriting : public mlir::OpConversionPattern<SimpleOp> {
  using base = typename mlir::OpConversionPattern<SimpleOp>;
  using base::OpConversionPattern;
  virtual mlir::LogicalResult
  matchAndRewrite(SimpleOp op, typename base::OpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    llvm::SmallVector<mlir::Type> resTys;
    if (mlir::failed(this->getTypeConverter()->convertTypes(
            op->getResultTypes(), resTys)))
      return mlir::failure();
    rewriter.replaceOpWithNewOp<SimpleOp>(op, resTys, adaptor.getOperands());
    return mlir::success();
  }
};

/// Fill the RewritePatternSet with our rewrite patterns
void populateFixedPtToArithConversionPatterns(mlir::RewritePatternSet &patterns,
                                              mlir::TypeConverter &converter) {
  // clang-format off
  patterns.add<AddOpLowering,
               SubOpLowering,
               ConstantOpLowering,
               BitcastOpLowering,
               MulOpLowering,
               DivOpLowering,
               ConvertOpLowering,
               FuncOpRewriting,
               SelectOpLowering,
               SimpleOpRewriting<func::ReturnOp>,
               SimpleOpRewriting<memref::LoadOp>,
               SimpleOpRewriting<memref::StoreOp>,
               SimpleOpRewriting<LLVM::LoadOp>,
               SimpleOpRewriting<LLVM::StoreOp>
               >(converter, patterns.getContext());
  // clang-format on
}

void ConvertFixedPtToArithPass::runOnOperation() {
  mlir::RewritePatternSet patterns(&getContext());
  FixedPtToArithTarget target(getContext());
  FixedPtToArithTypeConverter typeConverter(&getContext());

  populateFixedPtToArithConversionPatterns(patterns, typeConverter);

  /// Apply our rewrite patterns and thats it
  if (mlir::failed(mlir::applyPartialConversion(getOperation(), target,
                                                std::move(patterns))))
    signalPassFailure();
}

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
archgen::fixedpt::createConvertFixedPtToArithPass() {
  return std::make_unique<ConvertFixedPtToArithPass>();
}
