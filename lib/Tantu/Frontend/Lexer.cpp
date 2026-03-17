#include "Tantu/Frontend/Lexer.h"

#include <cctype>

// TokenKind

std::string_view tokenKindName(TokenKind kind) {
  switch (kind) {
  case LEFT_PAREN:          return "(";
  case RIGHT_PAREN:         return ")";
  case COMMA:               return ",";
  case LEFT_ANGLE:          return "<";
  case RIGHT_ANGLE:         return ">";
  case COLON:               return ":";
  case SEMICOLON:           return ";";
  case LEFT_CURLY_BRACKET:  return "{";
  case RIGHT_CURLY_BRACKET: return "}";
  case LEFT_BRACKET:        return "[";
  case RIGHT_BRACKET:       return "]";
  case EQUAL:               return "=";
  case IDENTIFIER:          return "identifier";
  case NUMBER:              return "number";
  case FN:                  return "fn";
  case LET:                 return "let";
  case TENSOR:              return "tensor";
  case ARROW_RIGHT:         return "->";
  case FLOAT_TYPE:          return "f32";
  case INTEGER_TYPE:        return "i32";
  case CONST:               return "const";
  case END_FILE:            return "<eof>";
  }
}

// Token

Token Token::Identifier(std::string_view name) {
  return {TokenKind::IDENTIFIER, name};
}

Token Token::Number(std::string_view val) { return {TokenKind::NUMBER, val}; }

Token Token::Simple(TokenKind k) { return {k, std::monostate{}}; }

std::string_view Token::getValue() const {
  return std::get<std::string_view>(value);
}

// Lexer

Lexer::Lexer(const std::string &input) : input(input), pos(0) {}

char Lexer::peek() const { return pos < input.size() ? input[pos] : '\0'; }

char Lexer::next() {
  if (pos < input.size())
    return input[pos++];
  return '\0';
}

void Lexer::skipWhitespace() {
  while (peek() == ' ' || peek() == '\t' || peek() == '\n')
    next();
}

void Lexer::skipComment() {
  while (peek() != '\n')
    next();
  next();
}

std::string_view Lexer::consumeIdentifier() {
  size_t start = pos - 1;
  while (peek() != '\0' &&
         (isalpha(peek()) || isdigit(peek()) || peek() == '_'))
    next();
  return std::string_view(input).substr(start, pos - start);
}

std::string_view Lexer::consumeNumber() {
  size_t start = pos - 1;
  while (peek() != '\0' && (isdigit(peek()) || peek() == '.'))
    next();
  return std::string_view(input).substr(start, pos - start);
}

Token Lexer::nextToken() {
  skipWhitespace();
  char ch = next();

  if (ch == '\0') {
    return Token::Simple(END_FILE);
  } else if (ch == '(') {
    return Token::Simple(LEFT_PAREN);
  } else if (ch == ')') {
    return Token::Simple(RIGHT_PAREN);
  } else if (ch == ',') {
    return Token::Simple(COMMA);
  } else if (ch == '<') {
    return Token::Simple(LEFT_ANGLE);
  } else if (ch == '>') {
    return Token::Simple(RIGHT_ANGLE);
  } else if (ch == ':') {
    return Token::Simple(COLON);
  } else if (ch == ';') {
    return Token::Simple(SEMICOLON);
  } else if (ch == '{') {
    return Token::Simple(LEFT_CURLY_BRACKET);
  } else if (ch == '}') {
    return Token::Simple(RIGHT_CURLY_BRACKET);
  } else if (ch == '[') {
    return Token::Simple(LEFT_BRACKET);
  } else if (ch == ']') {
    return Token::Simple(RIGHT_BRACKET);
  } else if (ch == '=') {
    return Token::Simple(EQUAL);
  } else if (ch == '-' && peek() == '-') {
    skipComment();
    return nextToken();
  } else if (ch == '-' && peek() == '>') {
    next();
    return Token::Simple(ARROW_RIGHT);
  }

  if (isdigit(ch)) {
    return Token::Number(consumeNumber());
  }

  std::string_view sv = consumeIdentifier();
  if (sv == "fn") {
    return Token::Simple(FN);
  } else if (sv == "let") {
    return Token::Simple(LET);
  } else if (sv == "tensor") {
    return Token::Simple(TENSOR);
  } else if (sv == "f32") {
    return Token::Simple(FLOAT_TYPE);
  } else if (sv == "i32") {
    return Token::Simple(INTEGER_TYPE);
  } else if (sv == "const") {
    return Token::Simple(CONST);
  } else {
    return Token::Identifier(sv);
  }
}

std::vector<Token> Lexer::parseTokens() {
  std::vector<Token> tokens;
  while (true) {
    Token token = nextToken();
    tokens.push_back(token);
    if (token.kind == END_FILE) {
      break;
    }
  }
  return tokens;
}
