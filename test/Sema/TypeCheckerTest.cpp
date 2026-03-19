#include "Tantu/Sema/TypeChecker.h"
#include "Tantu/Frontend/Lexer.h"
#include "Tantu/Frontend/Parser.h"
#include "llvm/Support/Error.h"
#include <gtest/gtest.h>

static bool typeChecks(const char *src) {
  std::string s(src);
  Lexer lexer(s);
  Parser parser(std::move(lexer));
  auto prog = parser.parseProgram();
  if (!prog) {
    llvm::handleAllErrors(prog.takeError(), [](const llvm::StringError &e) {
      std::cerr << "[parse error] " << e.getMessage() << "\n";
    });
    return false;
  }
  TypeChecker tc;
  llvm::Error err = tc.check((*prog).get());
  if (err) {
    llvm::handleAllErrors(std::move(err), [](const llvm::StringError &e) {
      std::cerr << "[type error] " << e.getMessage() << "\n";
    });
    return false;
  }
  return true;
}

static bool typeCheckFails(const char *src) {
  std::string s(src);
  Lexer lexer(s);
  Parser parser(std::move(lexer));
  auto prog = parser.parseProgram();
  if (!prog) {
    llvm::consumeError(prog.takeError());
    return false;
  }
  TypeChecker tc;
  llvm::Error err = tc.check((*prog).get());
  if (err) {
    llvm::handleAllErrors(std::move(err), [](const llvm::StringError &e) {
      std::cerr << "[type error] " << e.getMessage() << "\n";
    });
    return true;
  }
  return false;
}

TEST(TypeCheckerTest, EmptyProgram) { ASSERT_TRUE(typeChecks("")); }

TEST(TypeCheckerTest, ConstI32) {
  ASSERT_TRUE(typeChecks("const N: i32 = 4;"));
}

TEST(TypeCheckerTest, ConstF32) {
  ASSERT_TRUE(typeChecks("const SCALE: f32 = 1;"));
}

TEST(TypeCheckerTest, IdentityF32) {
  ASSERT_TRUE(typeChecks("fn id(x: f32) -> f32 { x }"));
}

TEST(TypeCheckerTest, IdentityI32) {
  ASSERT_TRUE(typeChecks("fn id(n: i32) -> i32 { n }"));
}

TEST(TypeCheckerTest, IdentityTensor) {
  ASSERT_TRUE(typeChecks("fn id(x: tensor<4>) -> tensor<4> { x }"));
}

TEST(TypeCheckerTest, IdentityTensorRank2) {
  ASSERT_TRUE(typeChecks("fn id(x: tensor<4,4>) -> tensor<4,4> { x }"));
}

TEST(TypeCheckerTest, AddMatchingShapes) {
  ASSERT_TRUE(typeChecks(
      "fn f(a: tensor<4>, b: tensor<4>) -> tensor<4> { add(a, b) }"));
}

TEST(TypeCheckerTest, SubMatchingShapes) {
  ASSERT_TRUE(typeChecks(
      "fn f(a: tensor<8>, b: tensor<8>) -> tensor<8> { sub(a, b) }"));
}

TEST(TypeCheckerTest, MulMatchingShapes) {
  ASSERT_TRUE(typeChecks(
      "fn f(a: tensor<4>, b: tensor<4>) -> tensor<4> { mul(a, b) }"));
}

TEST(TypeCheckerTest, DivMatchingShapes) {
  ASSERT_TRUE(typeChecks(
      "fn f(a: tensor<4>, b: tensor<4>) -> tensor<4> { div(a, b) }"));
}

TEST(TypeCheckerTest, NegTensor) {
  ASSERT_TRUE(typeChecks("fn f(x: tensor<4>) -> tensor<4> { neg(x) }"));
}

TEST(TypeCheckerTest, ExpTensor) {
  ASSERT_TRUE(typeChecks("fn f(x: tensor<4>) -> tensor<4> { exp(x) }"));
}

TEST(TypeCheckerTest, MaxElementwiseMatchingShapes) {
  ASSERT_TRUE(typeChecks(
      "fn f(a: tensor<4>, b: tensor<4>) -> tensor<4> { max(a, b) }"));
}

TEST(TypeCheckerTest, AddScalar) {
  ASSERT_TRUE(
      typeChecks("fn f(x: tensor<4>) -> tensor<4> { add_scalar(x, 1.0) }"));
}

TEST(TypeCheckerTest, SubScalar) {
  ASSERT_TRUE(
      typeChecks("fn f(x: tensor<4>) -> tensor<4> { sub_scalar(x, 1.0) }"));
}

TEST(TypeCheckerTest, MulScalar) {
  ASSERT_TRUE(
      typeChecks("fn f(x: tensor<4>) -> tensor<4> { mul_scalar(x, 2.0) }"));
}

TEST(TypeCheckerTest, DivScalar) {
  ASSERT_TRUE(
      typeChecks("fn f(x: tensor<4>) -> tensor<4> { div_scalar(x, 2.0) }"));
}

TEST(TypeCheckerTest, MaxScalar) {
  ASSERT_TRUE(
      typeChecks("fn f(x: tensor<4>) -> tensor<4> { max_scalar(x, 0.0) }"));
}

TEST(TypeCheckerTest, SumReturnsF32) {
  ASSERT_TRUE(typeChecks("fn f(x: tensor<4>) -> f32 { sum(x) }"));
}

TEST(TypeCheckerTest, MaxReduceReturnsF32) {
  ASSERT_TRUE(typeChecks("fn f(x: tensor<4>) -> f32 { max_reduce(x) }"));
}

TEST(TypeCheckerTest, SizeReturnsI32) {
  ASSERT_TRUE(typeChecks("fn f(x: tensor<4>) -> i32 { size(x) }"));
}

TEST(TypeCheckerTest, SizeOnRank2Tensor) {
  ASSERT_TRUE(typeChecks("fn f(x: tensor<4,4>) -> i32 { size(x) }"));
}

TEST(TypeCheckerTest, MatmulCompatibleDims) {
  ASSERT_TRUE(typeChecks(
      "fn f(a: tensor<2,3>, b: tensor<3,4>) -> tensor<2,4> { matmul(a, b) }"));
}

TEST(TypeCheckerTest, MatmulSquare) {
  ASSERT_TRUE(typeChecks(
      "fn f(a: tensor<4,4>, b: tensor<4,4>) -> tensor<4,4> { matmul(a, b) }"));
}

TEST(TypeCheckerTest, TransposeRank2) {
  ASSERT_TRUE(typeChecks(
      "fn f(x: tensor<4,8>) -> tensor<8,4> { transpose(x, [1, 0]) }"));
}

TEST(TypeCheckerTest, TransposeRank3) {
  ASSERT_TRUE(typeChecks(
      "fn f(x: tensor<4,8,16>) -> tensor<8,4,16> { transpose(x, [1, 0, 2]) }"));
}

TEST(TypeCheckerTest, CastI32ToF32) {
  ASSERT_TRUE(typeChecks("fn f(n: i32) -> f32 { cast(n) }"));
}

TEST(TypeCheckerTest, TensorLiteralRank1) {
  ASSERT_TRUE(typeChecks("fn f() -> tensor<3> { tensor<3>(1.0, 2.0, 3.0) }"));
}

TEST(TypeCheckerTest, TensorLiteralRank2) {
  ASSERT_TRUE(
      typeChecks("fn f() -> tensor<2,2> { tensor<2,2>(1.0, 2.0, 3.0, 4.0) }"));
}

TEST(TypeCheckerTest, SingleLetBinding) {
  ASSERT_TRUE(
      typeChecks("fn f(x: tensor<4>) -> tensor<4> { let y = neg(x); y }"));
}

TEST(TypeCheckerTest, LetBindingChain) {
  ASSERT_TRUE(typeChecks("fn f(x: tensor<4>) -> tensor<4> {"
                         "  let a = neg(x);"
                         "  let b = neg(a);"
                         "  b"
                         "}"));
}

TEST(TypeCheckerTest, LetBindingUsedInCall) {
  ASSERT_TRUE(typeChecks("fn f(x: tensor<4>) -> tensor<4> {"
                         "  let a = neg(x);"
                         "  add(x, a)"
                         "}"));
}

TEST(TypeCheckerTest, TwoFunctions) {
  ASSERT_TRUE(
      typeChecks("fn neg_all(x: tensor<4>) -> tensor<4> { neg(x) }"
                 "fn double_neg(x: tensor<4>) -> tensor<4> { neg_all(x) }"));
}

TEST(TypeCheckerTest, ConstThenFunction) {
  ASSERT_TRUE(typeChecks("const N: i32 = 4;"
                         "fn id(x: f32) -> f32 { x }"));
}

TEST(TypeCheckerTest, SoftmaxFromSpec) {
  ASSERT_TRUE(typeChecks("fn softmax(x: tensor<8>) -> tensor<8> {"
                         "  let m       = max_reduce(x);"
                         "  let shifted = sub_scalar(x, m);"
                         "  let exps    = exp(shifted);"
                         "  let total   = sum(exps);"
                         "  div_scalar(exps, total)"
                         "}"));
}

TEST(TypeCheckerTest, ReLUFromSpec) {
  ASSERT_TRUE(
      typeChecks("fn relu(x: tensor<4>) -> tensor<4> { max_scalar(x, 0.0) }"));
}

TEST(TypeCheckerTest, SwapBatchSeqFromSpec) {
  ASSERT_TRUE(
      typeChecks("fn swap_batch_seq(x: tensor<4,8,16>) -> tensor<8,4,16> {"
                 "  transpose(x, [1, 0, 2])"
                 "}"));
}

TEST(TypeCheckerTest, ErrorReturnTensorDeclaredF32) {
  ASSERT_TRUE(typeCheckFails("fn f(x: tensor<4>) -> f32 { x }"));
}

TEST(TypeCheckerTest, ErrorReturnF32DeclaredTensor) {
  ASSERT_TRUE(typeCheckFails("fn f(x: f32) -> tensor<4> { x }"));
}

TEST(TypeCheckerTest, ErrorReturnWrongTensorShape) {
  ASSERT_TRUE(typeCheckFails("fn f(x: tensor<4>) -> tensor<8> { x }"));
}

TEST(TypeCheckerTest, ErrorReturnI32DeclaredF32) {
  ASSERT_TRUE(typeCheckFails("fn f(x: tensor<4>) -> f32 { size(x) }"));
}

TEST(TypeCheckerTest, ErrorReturnF32DeclaredI32) {
  ASSERT_TRUE(typeCheckFails("fn f(x: tensor<4>) -> i32 { sum(x) }"));
}

TEST(TypeCheckerTest, ErrorReturnF32DeclaredTensorViaSum) {
  ASSERT_TRUE(typeCheckFails("fn f(x: tensor<4>) -> tensor<4> { sum(x) }"));
}

TEST(TypeCheckerTest, ErrorAddShapeMismatch) {
  ASSERT_TRUE(typeCheckFails(
      "fn f(a: tensor<4>, b: tensor<8>) -> tensor<4> { add(a, b) }"));
}

TEST(TypeCheckerTest, ErrorSubShapeMismatch) {
  ASSERT_TRUE(typeCheckFails(
      "fn f(a: tensor<4,4>, b: tensor<4,8>) -> tensor<4,4> { sub(a, b) }"));
}

TEST(TypeCheckerTest, ErrorMulShapeMismatch) {
  ASSERT_TRUE(typeCheckFails(
      "fn f(a: tensor<2,3>, b: tensor<3,2>) -> tensor<2,3> { mul(a, b) }"));
}

TEST(TypeCheckerTest, ErrorAddRankMismatch) {
  ASSERT_TRUE(typeCheckFails(
      "fn f(a: tensor<4>, b: tensor<2,2>) -> tensor<4> { add(a, b) }"));
}

TEST(TypeCheckerTest, ErrorMatmulIncompatibleInnerDims) {
  ASSERT_TRUE(typeCheckFails(
      "fn f(a: tensor<2,3>, b: tensor<4,5>) -> tensor<2,5> { matmul(a, b) }"));
}

TEST(TypeCheckerTest, ErrorMatmulWrongResultShape) {
  ASSERT_TRUE(typeCheckFails(
      "fn f(a: tensor<2,3>, b: tensor<3,4>) -> tensor<2,3> { matmul(a, b) }"));
}

TEST(TypeCheckerTest, ErrorTransposePermTooShort) {
  ASSERT_TRUE(typeCheckFails(
      "fn f(x: tensor<4,8>) -> tensor<8,4> { transpose(x, [0]) }"));
}

TEST(TypeCheckerTest, ErrorTransposePermTooLong) {
  ASSERT_TRUE(typeCheckFails(
      "fn f(x: tensor<4,8>) -> tensor<8,4> { transpose(x, [1, 0, 0]) }"));
}

TEST(TypeCheckerTest, ErrorTransposePermOutOfRange) {
  ASSERT_TRUE(typeCheckFails(
      "fn f(x: tensor<4,8>) -> tensor<8,4> { transpose(x, [2, 0]) }"));
}

TEST(TypeCheckerTest, ErrorTransposePermDuplicateIndex) {
  ASSERT_TRUE(typeCheckFails(
      "fn f(x: tensor<4,8>) -> tensor<4,8> { transpose(x, [0, 0]) }"));
}

TEST(TypeCheckerTest, ErrorTransposeWrongResultShape) {
  ASSERT_TRUE(typeCheckFails(
      "fn f(x: tensor<4,8>) -> tensor<4,8> { transpose(x, [1, 0]) }"));
}

TEST(TypeCheckerTest, ErrorTensorLiteralTooFewArgs) {
  ASSERT_TRUE(typeCheckFails("fn f() -> tensor<3> { tensor<3>(1.0, 2.0) }"));
}

TEST(TypeCheckerTest, ErrorTensorLiteralTooManyArgs) {
  ASSERT_TRUE(
      typeCheckFails("fn f() -> tensor<3> { tensor<3>(1.0, 2.0, 3.0, 4.0) }"));
}

TEST(TypeCheckerTest, ErrorTensorLiteralRank2TooFewArgs) {
  ASSERT_TRUE(typeCheckFails(
      "fn f() -> tensor<2,3> { tensor<2,3>(1.0, 2.0, 3.0, 4.0) }"));
}

TEST(TypeCheckerTest, ErrorUseUndefinedName) {
  ASSERT_TRUE(typeCheckFails("fn f(x: tensor<4>) -> tensor<4> { y }"));
}

TEST(TypeCheckerTest, ErrorRebindName) {
  ASSERT_TRUE(typeCheckFails("fn f(x: tensor<4>) -> tensor<4> {"
                             "  let y = neg(x);"
                             "  let y = neg(x);"
                             "  y"
                             "}"));
}

TEST(TypeCheckerTest, ErrorUseLetNameBeforeBinding) {
  ASSERT_TRUE(typeCheckFails("fn f(x: tensor<4>) -> tensor<4> {"
                             "  let a = add(x, b);"
                             "  let b = neg(x);"
                             "  a"
                             "}"));
}

TEST(TypeCheckerTest, ErrorScalarArgWhereVectorExpected) {
  ASSERT_TRUE(typeCheckFails("fn f(x: f32) -> f32 { neg(x) }"));
}

TEST(TypeCheckerTest, ErrorTensorArgWhereScalarExpected) {
  ASSERT_TRUE(typeCheckFails(
      "fn f(a: tensor<4>, b: tensor<4>) -> tensor<4> { add_scalar(a, b) }"));
}

TEST(TypeCheckerTest, ErrorSumOnScalar) {
  ASSERT_TRUE(typeCheckFails("fn f(x: f32) -> f32 { sum(x) }"));
}

TEST(TypeCheckerTest, ErrorMaxReduceOnScalar) {
  ASSERT_TRUE(typeCheckFails("fn f(x: f32) -> f32 { max_reduce(x) }"));
}

TEST(TypeCheckerTest, ErrorSizeOnScalar) {
  ASSERT_TRUE(typeCheckFails("fn f(x: f32) -> i32 { size(x) }"));
}
