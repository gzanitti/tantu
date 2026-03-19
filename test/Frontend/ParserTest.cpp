#include "Tantu/Frontend/Parser.h"
#include "Tantu/Frontend/Lexer.h"
#include "Tantu/Frontend/PrettyPrinter.h"
#include "llvm/Support/Error.h"
#include <gtest/gtest.h>
#include <sstream>

static std::string parseAndPrint(const char *src) {
  std::string s(src);
  Lexer lexer(s);
  Parser parser(std::move(lexer));
  auto prog = parser.parseProgram();
  if (!prog) {
    std::string msg;
    llvm::handleAllErrors(prog.takeError(), [&](const llvm::StringError &e) {
      msg = e.getMessage();
    });
    return "<parse error: " + msg + ">";
  }
  std::ostringstream oss;
  auto *old = std::cout.rdbuf(oss.rdbuf());
  PrettyPrinter pp;
  (*prog)->accept(pp);
  std::cout.rdbuf(old);
  return oss.str();
}

static bool parseFails(const char *src) {
  std::string s(src);
  Lexer lexer(s);
  Parser parser(std::move(lexer));
  auto prog = parser.parseProgram();
  if (!prog) {
    llvm::handleAllErrors(prog.takeError(), [&](const llvm::StringError &e) {
      std::cerr << "[parse error] " << e.getMessage() << "\n";
    });
    return true;
  }
  return false;
}

TEST(ParserTest, EmptyProgram) {
  ASSERT_EQ(parseAndPrint(""), "Program(\n)\n");
}

TEST(ParserTest, ConstDefI32) {
  ASSERT_EQ(parseAndPrint("const VALUE: i32 = 8;"), "Program(\n"
                                                    "ConstDef(VALUE 8 : i32)\n"
                                                    ")\n");
}

TEST(ParserTest, ConstDefF32) {
  ASSERT_EQ(parseAndPrint("const SCALE: f32 = 1;"), "Program(\n"
                                                    "ConstDef(SCALE 1 : f32)\n"
                                                    ")\n");
}

TEST(ParserTest, IdentityFunctionF32) {
  ASSERT_EQ(parseAndPrint("fn id(x: f32) -> f32 { x }"),
            "Program(\n"
            "FunctionDef(id \n"
            "  Params: Param(x : f32)\n"
            "  Body:   Return: x : f32)\n"
            ")\n");
}

TEST(ParserTest, IdentityFunctionI32) {
  ASSERT_EQ(parseAndPrint("fn id(n: i32) -> i32 { n }"),
            "Program(\n"
            "FunctionDef(id \n"
            "  Params: Param(n : i32)\n"
            "  Body:   Return: n : i32)\n"
            ")\n");
}

TEST(ParserTest, IdentityFunctionTensorRank1) {
  ASSERT_EQ(parseAndPrint("fn id(x: tensor<4>) -> tensor<4> { x }"),
            "Program(\n"
            "FunctionDef(id \n"
            "  Params: Param(x : Tensor<4>)\n"
            "  Body:   Return: x : Tensor<4>)\n"
            ")\n");
}

TEST(ParserTest, IdentityFunctionTensorRank2) {
  ASSERT_EQ(parseAndPrint("fn id(x: tensor<4,4>) -> tensor<4,4> { x }"),
            "Program(\n"
            "FunctionDef(id \n"
            "  Params: Param(x : Tensor<4x4>)\n"
            "  Body:   Return: x : Tensor<4x4>)\n"
            ")\n");
}

TEST(ParserTest, IdentityFunctionTensorRank3) {
  ASSERT_EQ(parseAndPrint("fn id(x: tensor<2,3,4>) -> tensor<2,3,4> { x }"),
            "Program(\n"
            "FunctionDef(id \n"
            "  Params: Param(x : Tensor<2x3x4>)\n"
            "  Body:   Return: x : Tensor<2x3x4>)\n"
            ")\n");
}

TEST(ParserTest, FunctionNoParams) {
  ASSERT_EQ(parseAndPrint("fn zero() -> f32 { 0.0 }"),
            "Program(\n"
            "FunctionDef(zero \n"
            "  Params:   Body:   Return: 0 : f32)\n"
            ")\n");
}

TEST(ParserTest, FunctionTwoScalarParams) {
  ASSERT_EQ(parseAndPrint("fn f(a: f32, b: f32) -> f32 { a }"),
            "Program(\n"
            "FunctionDef(f \n"
            "  Params: Param(a : f32)\n"
            "Param(b : f32)\n"
            "  Body:   Return: a : f32)\n"
            ")\n");
}

TEST(ParserTest, FunctionTwoTensorParams) {
  ASSERT_EQ(
      parseAndPrint("fn wrap(a: tensor<2,3>, b: tensor<3,4>) -> tensor<2,4> {"
                    " matmul(a, b) }"),
      "Program(\n"
      "FunctionDef(wrap \n"
      "  Params: Param(a : Tensor<2x3>)\n"
      "Param(b : Tensor<3x4>)\n"
      "  Body:   Return: matmul(a, b) : Tensor<2x4>)\n"
      ")\n");
}

TEST(ParserTest, ElementwiseAdd) {
  ASSERT_EQ(parseAndPrint(
                "fn f(a: tensor<4>, b: tensor<4>) -> tensor<4> { add(a, b) }"),
            "Program(\n"
            "FunctionDef(f \n"
            "  Params: Param(a : Tensor<4>)\n"
            "Param(b : Tensor<4>)\n"
            "  Body:   Return: add(a, b) : Tensor<4>)\n"
            ")\n");
}

TEST(ParserTest, NegCall) {
  ASSERT_EQ(parseAndPrint("fn f(x: tensor<4>) -> tensor<4> { neg(x) }"),
            "Program(\n"
            "FunctionDef(f \n"
            "  Params: Param(x : Tensor<4>)\n"
            "  Body:   Return: neg(x) : Tensor<4>)\n"
            ")\n");
}

TEST(ParserTest, MulScalarCall) {
  ASSERT_EQ(
      parseAndPrint("fn f(x: tensor<4>) -> tensor<4> { mul_scalar(x, 2.0) }"),
      "Program(\n"
      "FunctionDef(f \n"
      "  Params: Param(x : Tensor<4>)\n"
      "  Body:   Return: mul_scalar(x, 2) : Tensor<4>)\n"
      ")\n");
}

TEST(ParserTest, SumReduction) {
  ASSERT_EQ(parseAndPrint("fn f(x: tensor<4>) -> f32 { sum(x) }"),
            "Program(\n"
            "FunctionDef(f \n"
            "  Params: Param(x : Tensor<4>)\n"
            "  Body:   Return: sum(x) : f32)\n"
            ")\n");
}

TEST(ParserTest, MaxReduceCall) {
  ASSERT_EQ(parseAndPrint("fn f(x: tensor<4>) -> f32 { max_reduce(x) }"),
            "Program(\n"
            "FunctionDef(f \n"
            "  Params: Param(x : Tensor<4>)\n"
            "  Body:   Return: max_reduce(x) : f32)\n"
            ")\n");
}

TEST(ParserTest, SizeCall) {
  ASSERT_EQ(parseAndPrint("fn f(x: tensor<4>) -> i32 { size(x) }"),
            "Program(\n"
            "FunctionDef(f \n"
            "  Params: Param(x : Tensor<4>)\n"
            "  Body:   Return: size(x) : i32)\n"
            ")\n");
}

TEST(ParserTest, TensorLiteralRank1) {
  ASSERT_EQ(parseAndPrint("fn f() -> tensor<3> { tensor<3>(1.0, 2.0, 3.0) }"),
            "Program(\n"
            "FunctionDef(f \n"
            "  Params:   Body:   Return: Tensor<3>[1, 2, 3] : Tensor<3>)\n"
            ")\n");
}

TEST(ParserTest, TensorLiteralRank2) {
  ASSERT_EQ(
      parseAndPrint(
          "fn f() -> tensor<2,2> { tensor<2,2>(1.0, 2.0, 3.0, 4.0) }"),
      "Program(\n"
      "FunctionDef(f \n"
      "  Params:   Body:   Return: Tensor<2x2>[1, 2, 3, 4] : Tensor<2x2>)\n"
      ")\n");
}

TEST(ParserTest, SingleLetBinding) {
  ASSERT_EQ(
      parseAndPrint("fn f(x: tensor<4>) -> tensor<4> { let y = neg(x); y }"),
      "Program(\n"
      "FunctionDef(f \n"
      "  Params: Param(x : Tensor<4>)\n"
      "  Body: LetBinding(y = neg(x));\n"
      "  Return: y : Tensor<4>)\n"
      ")\n");
}

TEST(ParserTest, ReLUFromSpec) {
  ASSERT_EQ(parseAndPrint(
                "fn relu(x: tensor<4>) -> tensor<4> { max_scalar(x, 0.0) }"),
            "Program(\n"
            "FunctionDef(relu \n"
            "  Params: Param(x : Tensor<4>)\n"
            "  Body:   Return: max_scalar(x, 0) : Tensor<4>)\n"
            ")\n");
}

TEST(ParserTest, SoftmaxFromSpec) {
  const char *src = "fn softmax(x: tensor<8>) -> tensor<8> {"
                    "  let m       = max_reduce(x);"
                    "  let shifted = sub_scalar(x, m);"
                    "  let exps    = exp(shifted);"
                    "  let total   = sum(exps);"
                    "  div_scalar(exps, total)"
                    "}";

  ASSERT_EQ(parseAndPrint(src),
            "Program(\n"
            "FunctionDef(softmax \n"
            "  Params: Param(x : Tensor<8>)\n"
            "  Body: LetBinding(m = max_reduce(x));\n"
            "LetBinding(shifted = sub_scalar(x, m));\n"
            "LetBinding(exps = exp(shifted));\n"
            "LetBinding(total = sum(exps));\n"
            "  Return: div_scalar(exps, total) : Tensor<8>)\n"
            ")\n");
}

TEST(ParserTest, ConstThenFunction) {
  ASSERT_EQ(parseAndPrint("const N: i32 = 4; fn id(x: f32) -> f32 { x }"),
            "Program(\n"
            "ConstDef(N 4 : i32)\n"
            "FunctionDef(id \n"
            "  Params: Param(x : f32)\n"
            "  Body:   Return: x : f32)\n"
            ")\n");
}

TEST(ParserTest, TwoFunctions) {
  ASSERT_EQ(parseAndPrint("fn neg_all(x: tensor<4>) -> tensor<4> { neg(x) }"
                          "fn f(x: tensor<4>) -> tensor<4> { neg_all(x) }"),
            "Program(\n"
            "FunctionDef(neg_all \n"
            "  Params: Param(x : Tensor<4>)\n"
            "  Body:   Return: neg(x) : Tensor<4>)\n"
            "FunctionDef(f \n"
            "  Params: Param(x : Tensor<4>)\n"
            "  Body:   Return: neg_all(x) : Tensor<4>)\n"
            ")\n");
}

TEST(ParserTest, TransposeWithPermutation) {
  ASSERT_EQ(
      parseAndPrint(
          "fn swap_batch_seq(x: tensor<4,8,16>) -> tensor<8,4,16> {"
          "  transpose(x, [1, 0, 2])"
          "}"),
      "Program(\n"
      "FunctionDef(swap_batch_seq \n"
      "  Params: Param(x : Tensor<4x8x16>)\n"
      "  Body:   Return: transpose(x, Permutation(1 , 0 , 2 )) : Tensor<8x4x16>)\n"
      ")\n");
}

TEST(ParserTest, ErrorPermutationNonInteger) {
  ASSERT_TRUE(parseFails(
      "fn f(x: tensor<4,4>) -> tensor<4,4> { transpose(x, [0, x]) }"));
}

TEST(ParserTest, ErrorBareNumberAtTopLevel) { ASSERT_TRUE(parseFails("42")); }

TEST(ParserTest, ErrorEmptyFunctionBody) {
  ASSERT_TRUE(parseFails("fn f(x: f32) -> f32 { }"));
}

TEST(ParserTest, ErrorLetMissingSemicolon) {
  ASSERT_TRUE(parseFails("fn f(x: f32) -> f32 { let y = x y }"));
}

TEST(ParserTest, ErrorMissingClosingBrace) {
  ASSERT_TRUE(parseFails("fn f(x: f32) -> f32 { x"));
}

TEST(ParserTest, ErrorMissingArrow) {
  ASSERT_TRUE(parseFails("fn f(x: f32) f32 { x }"));
}

TEST(ParserTest, ErrorMissingReturnType) {
  ASSERT_TRUE(parseFails("fn f(x: f32) -> { x }"));
}

TEST(ParserTest, ErrorConstMissingSemicolon) {
  ASSERT_TRUE(parseFails("const N: i32 = 8"));
}
