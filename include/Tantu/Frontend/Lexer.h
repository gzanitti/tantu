#pragma once

#include <string>
#include <variant>
#include <vector>

enum TokenKind {
  LEFT_PAREN,
  RIGHT_PAREN,
  COMMA,
  LEFT_ANGLE,  // <
  RIGHT_ANGLE, // >
  COLON,
  SEMICOLON,
  LEFT_CURLY_BRACKET,  // {
  RIGHT_CURLY_BRACKET, // }
  LEFT_BRACKET,        // [
  RIGHT_BRACKET,       // ]
  EQUAL,               // =

  IDENTIFIER,
  NUMBER,

  FN,
  LET,
  TENSOR,
  ARROW_RIGHT,  // ->
  FLOAT_TYPE,   // f32
  INTEGER_TYPE, // i32
  CONST,

  END_FILE,
};

std::string_view tokenKindName(TokenKind kind);

struct Token {
  TokenKind kind;
  std::variant<std::monostate, std::string_view> value;

  static Token Identifier(std::string_view name);
  static Token Number(std::string_view val);
  static Token Simple(TokenKind k);

  std::string_view getValue() const;
};

struct Lexer {
  Lexer(const std::string &input);

  std::string_view input;
  size_t pos;

  char peek() const;
  char next();
  void skipWhitespace();
  void skipComment();
  std::string_view consumeIdentifier();
  std::string_view consumeNumber();
  Token nextToken();
  std::vector<Token> parseTokens();
};
