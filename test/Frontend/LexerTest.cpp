#include "Tantu/Frontend/Lexer.h"
#include <gtest/gtest.h>
#include <memory>

struct Tokens {
  std::unique_ptr<std::string> src;
  std::vector<Token> v;
  size_t size() const { return v.size(); }
};

static Tokens lex(const char* s) {
  auto src = std::make_unique<std::string>(s);
  Lexer lexer(*src);
  auto tokens = lexer.parseTokens();
  return {std::move(src), std::move(tokens)};
}

TEST(LexerTest, EmptyInputGivesEOF) {
  auto t = lex("");
  ASSERT_EQ(t.size(), 1u);
  ASSERT_EQ(t.v[0].kind, END_FILE);
}

TEST(LexerTest, WhitespaceOnlyGivesEOF) {
  auto t = lex("   \t\n  ");
  ASSERT_EQ(t.size(), 1u);
  ASSERT_EQ(t.v[0].kind, END_FILE);
}

TEST(LexerTest, KeywordFn) {
  auto t = lex("fn");
  ASSERT_EQ(t.v[0].kind, FN);
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, KeywordLet) {
  auto t = lex("let");
  ASSERT_EQ(t.v[0].kind, LET);
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, KeywordConst) {
  auto t = lex("const");
  ASSERT_EQ(t.v[0].kind, CONST);
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, KeywordTensor) {
  auto t = lex("tensor");
  ASSERT_EQ(t.v[0].kind, TENSOR);
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, TypeF32) {
  auto t = lex("f32");
  ASSERT_EQ(t.v[0].kind, FLOAT_TYPE);
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, TypeI32) {
  auto t = lex("i32");
  ASSERT_EQ(t.v[0].kind, INTEGER_TYPE);
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, FnPrefixIsIdentifier) {
  auto t = lex("fnn");
  ASSERT_EQ(t.v[0].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "fnn");
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, LetPrefixIsIdentifier) {
  auto t = lex("letting");
  ASSERT_EQ(t.v[0].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "letting");
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, ConstPrefixIsIdentifier) {
  auto t = lex("constant");
  ASSERT_EQ(t.v[0].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "constant");
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, TensorPrefixIsIdentifier) {
  auto t = lex("tensors");
  ASSERT_EQ(t.v[0].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "tensors");
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, F32PrefixIsIdentifier) {
  auto t = lex("f32x");
  ASSERT_EQ(t.v[0].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "f32x");
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, I32PrefixIsIdentifier) {
  auto t = lex("i32k");
  ASSERT_EQ(t.v[0].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "i32k");
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, SimpleIdentifier) {
  auto t = lex("add");
  ASSERT_EQ(t.v[0].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "add");
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, SingleLetterIdentifier) {
  auto t = lex("x");
  ASSERT_EQ(t.v[0].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "x");
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, MultipleIdentifiersSpaceSeparated) {
  auto t = lex("foo bar baz");
  ASSERT_EQ(t.v[0].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "foo");
  ASSERT_EQ(t.v[1].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[1].value), "bar");
  ASSERT_EQ(t.v[2].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[2].value), "baz");
  ASSERT_EQ(t.v[3].kind, END_FILE);
}

TEST(LexerTest, IntegerLiteralIsNumber) {
  auto t = lex("42");
  ASSERT_EQ(t.v[0].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "42");
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, ZeroIsNumber) {
  auto t = lex("0");
  ASSERT_EQ(t.v[0].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "0");
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, FloatLiteralIsNumber) {
  auto t = lex("3.14");
  ASSERT_EQ(t.v[0].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "3.14");
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, ZeroPointZeroIsNumber) {
  auto t = lex("0.0");
  ASSERT_EQ(t.v[0].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "0.0");
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, NumberDoesNotAbsorbFollowingIdentifier) {
  auto t = lex("42add");
  ASSERT_EQ(t.v[0].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "42");
  ASSERT_EQ(t.v[1].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[1].value), "add");
  ASSERT_EQ(t.v[2].kind, END_FILE);
}

TEST(LexerTest, AllSymbolsInSequence) {
  auto t = lex("( ) , < > : ; { } [ ]");
  ASSERT_EQ(t.v[0].kind, LEFT_PAREN);
  ASSERT_EQ(t.v[1].kind, RIGHT_PAREN);
  ASSERT_EQ(t.v[2].kind, COMMA);
  ASSERT_EQ(t.v[3].kind, LEFT_ANGLE);
  ASSERT_EQ(t.v[4].kind, RIGHT_ANGLE);
  ASSERT_EQ(t.v[5].kind, COLON);
  ASSERT_EQ(t.v[6].kind, SEMICOLON);
  ASSERT_EQ(t.v[7].kind, LEFT_CURLY_BRACKET);
  ASSERT_EQ(t.v[8].kind, RIGHT_CURLY_BRACKET);
  ASSERT_EQ(t.v[9].kind, LEFT_BRACKET);
  ASSERT_EQ(t.v[10].kind, RIGHT_BRACKET);
  ASSERT_EQ(t.v[11].kind, END_FILE);
}

TEST(LexerTest, ArrowRight) {
  auto t = lex("->");
  ASSERT_EQ(t.v[0].kind, ARROW_RIGHT);
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, ArrowRightBetweenTypes) {
  auto t = lex("f32 -> f32");
  ASSERT_EQ(t.v[0].kind, FLOAT_TYPE);
  ASSERT_EQ(t.v[1].kind, ARROW_RIGHT);
  ASSERT_EQ(t.v[2].kind, FLOAT_TYPE);
  ASSERT_EQ(t.v[3].kind, END_FILE);
}

TEST(LexerTest, CommentLineIsFullySkipped) {
  auto t = lex("-- this is a comment\nfn");
  ASSERT_EQ(t.v[0].kind, FN);
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, CommentOnlyInput) {
  auto t = lex("-- just a comment\n");
  ASSERT_EQ(t.size(), 1u);
  ASSERT_EQ(t.v[0].kind, END_FILE);
}

TEST(LexerTest, InlineCommentBetweenTokens) {
  auto t = lex("fn -- function keyword\nlet");
  ASSERT_EQ(t.v[0].kind, FN);
  ASSERT_EQ(t.v[1].kind, LET);
  ASSERT_EQ(t.v[2].kind, END_FILE);
}

TEST(LexerTest, MultipleCommentLines) {
  auto t = lex("-- line 1\n-- line 2\nfn");
  ASSERT_EQ(t.v[0].kind, FN);
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, LeadingTrailingSpaces) {
  auto t = lex("  fn  ");
  ASSERT_EQ(t.v[0].kind, FN);
  ASSERT_EQ(t.v[1].kind, END_FILE);
}

TEST(LexerTest, TabsBetweenTokens) {
  auto t = lex("fn\tlet");
  ASSERT_EQ(t.v[0].kind, FN);
  ASSERT_EQ(t.v[1].kind, LET);
  ASSERT_EQ(t.v[2].kind, END_FILE);
}

TEST(LexerTest, NewlineBetweenTokens) {
  auto t = lex("fn\nlet");
  ASSERT_EQ(t.v[0].kind, FN);
  ASSERT_EQ(t.v[1].kind, LET);
  ASSERT_EQ(t.v[2].kind, END_FILE);
}

TEST(LexerTest, MixedWhitespaceBetweenTokens) {
  auto t = lex("fn \t\n let");
  ASSERT_EQ(t.v[0].kind, FN);
  ASSERT_EQ(t.v[1].kind, LET);
  ASSERT_EQ(t.v[2].kind, END_FILE);
}

TEST(LexerTest, TensorTypeWithSpaces) {
  auto t = lex("tensor < 4 , 4 >");
  ASSERT_EQ(t.v[0].kind, TENSOR);
  ASSERT_EQ(t.v[1].kind, LEFT_ANGLE);
  ASSERT_EQ(t.v[2].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[2].value), "4");
  ASSERT_EQ(t.v[3].kind, COMMA);
  ASSERT_EQ(t.v[4].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[4].value), "4");
  ASSERT_EQ(t.v[5].kind, RIGHT_ANGLE);
  ASSERT_EQ(t.v[6].kind, END_FILE);
}

TEST(LexerTest, TensorTypeNoSpaces) {
  auto t = lex("tensor<4,4>");
  ASSERT_EQ(t.v[0].kind, TENSOR);
  ASSERT_EQ(t.v[1].kind, LEFT_ANGLE);
  ASSERT_EQ(t.v[2].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[2].value), "4");
  ASSERT_EQ(t.v[3].kind, COMMA);
  ASSERT_EQ(t.v[4].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[4].value), "4");
  ASSERT_EQ(t.v[5].kind, RIGHT_ANGLE);
  ASSERT_EQ(t.v[6].kind, END_FILE);
}

TEST(LexerTest, TensorTypeRank1) {
  auto t = lex("tensor<8>");
  ASSERT_EQ(t.v[0].kind, TENSOR);
  ASSERT_EQ(t.v[1].kind, LEFT_ANGLE);
  ASSERT_EQ(t.v[2].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[2].value), "8");
  ASSERT_EQ(t.v[3].kind, RIGHT_ANGLE);
  ASSERT_EQ(t.v[4].kind, END_FILE);
}

TEST(LexerTest, TensorTypeRank3) {
  auto t = lex("tensor<2,3,4>");
  ASSERT_EQ(t.v[0].kind, TENSOR);
  ASSERT_EQ(t.v[1].kind, LEFT_ANGLE);
  ASSERT_EQ(t.v[2].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[2].value), "2");
  ASSERT_EQ(t.v[3].kind, COMMA);
  ASSERT_EQ(t.v[4].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[4].value), "3");
  ASSERT_EQ(t.v[5].kind, COMMA);
  ASSERT_EQ(t.v[6].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[6].value), "4");
  ASSERT_EQ(t.v[7].kind, RIGHT_ANGLE);
  ASSERT_EQ(t.v[8].kind, END_FILE);
}

TEST(LexerTest, TensorTypeI32Element) {
  auto t = lex("tensor<4>");
  ASSERT_EQ(t.v[0].kind, TENSOR);
  ASSERT_EQ(t.v[1].kind, LEFT_ANGLE);
  ASSERT_EQ(t.v[2].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[2].value), "4");
  ASSERT_EQ(t.v[3].kind, RIGHT_ANGLE);
  ASSERT_EQ(t.v[4].kind, END_FILE);
}

TEST(LexerTest, PermutationArrayLiteral) {
  auto t = lex("[1, 0, 2]");
  ASSERT_EQ(t.v[0].kind, LEFT_BRACKET);
  ASSERT_EQ(t.v[1].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[1].value), "1");
  ASSERT_EQ(t.v[2].kind, COMMA);
  ASSERT_EQ(t.v[3].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[3].value), "0");
  ASSERT_EQ(t.v[4].kind, COMMA);
  ASSERT_EQ(t.v[5].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[5].value), "2");
  ASSERT_EQ(t.v[6].kind, RIGHT_BRACKET);
  ASSERT_EQ(t.v[7].kind, END_FILE);
}

TEST(LexerTest, BasicFn) {
  auto t = lex("fn add(a: f32) -> f32 { a }");
  ASSERT_EQ(t.v[0].kind, FN);
  ASSERT_EQ(t.v[1].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[1].value), "add");
  ASSERT_EQ(t.v[2].kind, LEFT_PAREN);
  ASSERT_EQ(t.v[3].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[3].value), "a");
  ASSERT_EQ(t.v[4].kind, COLON);
  ASSERT_EQ(t.v[5].kind, FLOAT_TYPE);
  ASSERT_EQ(t.v[6].kind, RIGHT_PAREN);
  ASSERT_EQ(t.v[7].kind, ARROW_RIGHT);
  ASSERT_EQ(t.v[8].kind, FLOAT_TYPE);
  ASSERT_EQ(t.v[9].kind, LEFT_CURLY_BRACKET);
  ASSERT_EQ(t.v[10].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[10].value), "a");
  ASSERT_EQ(t.v[11].kind, RIGHT_CURLY_BRACKET);
  ASSERT_EQ(t.v[12].kind, END_FILE);
}

TEST(LexerTest, FunctionWithTwoParams) {
  auto t = lex("fn add(a: f32, b: f32) -> f32 { a }");
  ASSERT_EQ(t.v[0].kind, FN);
  ASSERT_EQ(t.v[1].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[1].value), "add");
  ASSERT_EQ(t.v[2].kind, LEFT_PAREN);
  ASSERT_EQ(t.v[3].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[3].value), "a");
  ASSERT_EQ(t.v[4].kind, COLON);
  ASSERT_EQ(t.v[5].kind, FLOAT_TYPE);
  ASSERT_EQ(t.v[6].kind, COMMA);
  ASSERT_EQ(t.v[7].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[7].value), "b");
  ASSERT_EQ(t.v[8].kind, COLON);
  ASSERT_EQ(t.v[9].kind, FLOAT_TYPE);
  ASSERT_EQ(t.v[10].kind, RIGHT_PAREN);
  ASSERT_EQ(t.v[11].kind, ARROW_RIGHT);
  ASSERT_EQ(t.v[12].kind, FLOAT_TYPE);
  ASSERT_EQ(t.v[13].kind, LEFT_CURLY_BRACKET);
  ASSERT_EQ(t.v[14].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[14].value), "a");
  ASSERT_EQ(t.v[15].kind, RIGHT_CURLY_BRACKET);
  ASSERT_EQ(t.v[16].kind, END_FILE);
}

TEST(LexerTest, ConstDefinitionHeader) {
  auto t = lex("const SCALE : f32");
  ASSERT_EQ(t.v[0].kind, CONST);
  ASSERT_EQ(t.v[1].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[1].value), "SCALE");
  ASSERT_EQ(t.v[2].kind, COLON);
  ASSERT_EQ(t.v[3].kind, FLOAT_TYPE);
  ASSERT_EQ(t.v[4].kind, END_FILE);
}

TEST(LexerTest, I32TypeInParam) {
  auto t = lex("n: i32");
  ASSERT_EQ(t.v[0].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "n");
  ASSERT_EQ(t.v[1].kind, COLON);
  ASSERT_EQ(t.v[2].kind, INTEGER_TYPE);
  ASSERT_EQ(t.v[3].kind, END_FILE);
}

TEST(LexerTest, CallExpressionWithIntegerArg) {
  auto t = lex("foo(42)");
  ASSERT_EQ(t.v[0].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "foo");
  ASSERT_EQ(t.v[1].kind, LEFT_PAREN);
  ASSERT_EQ(t.v[2].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[2].value), "42");
  ASSERT_EQ(t.v[3].kind, RIGHT_PAREN);
  ASSERT_EQ(t.v[4].kind, END_FILE);
}

TEST(LexerTest, CallExpressionWithFloatArg) {
  auto t = lex("foo(0.5)");
  ASSERT_EQ(t.v[0].kind, IDENTIFIER);
  ASSERT_EQ(std::get<std::string_view>(t.v[0].value), "foo");
  ASSERT_EQ(t.v[1].kind, LEFT_PAREN);
  ASSERT_EQ(t.v[2].kind, NUMBER);
  ASSERT_EQ(std::get<std::string_view>(t.v[2].value), "0.5");
  ASSERT_EQ(t.v[3].kind, RIGHT_PAREN);
  ASSERT_EQ(t.v[4].kind, END_FILE);
}
