#include "Tantu/Frontend/Parser.h"
#include "Tantu/Frontend/AST.h"
#include "Tantu/Frontend/Lexer.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>

Parser::Parser(Lexer lexer)
    : lexer(std::move(lexer)), current(this->lexer.nextToken()){};

Token Parser::next() {
  current = lexer.nextToken();
  return current;
};

llvm::Error Parser::makeError(std::string message) {
  return llvm::make_error<llvm::StringError>(message,
                                             llvm::inconvertibleErrorCode());
}

llvm::Expected<Token> Parser::expect_next(TokenKind kind) {
  next();
  if (current.kind != kind) {
    return makeError("expected '" + std::string(tokenKindName(kind)) +
                     "', got '" + std::string(tokenKindName(current.kind)) +
                     "'");
  }
  return current;
}

llvm::Expected<std::unique_ptr<Program>> Parser::parseProgram() {
  std::vector<std::unique_ptr<Definition>> defs;

  while (current.kind != END_FILE) {
    if (current.kind == CONST) {
      llvm::Expected<std::unique_ptr<Definition>> result = parseConstDef();
      if (!result)
        return result.takeError();
      auto constDef = std::move(*result);
      defs.push_back(std::move(constDef));
    } else if (current.kind == FN) {
      llvm::Expected<std::unique_ptr<Definition>> result = parseFunctionDef();
      if (!result)
        return result.takeError();
      auto functionDef = std::move(*result);
      defs.push_back(std::move(functionDef));
    } else {
      return makeError("expected '" + std::string(tokenKindName(CONST)) +
                       "' or '" + std::string(tokenKindName(FN)) + "', got '" +
                       std::string(tokenKindName(current.kind)) + "'");
    }
  }
  return std::make_unique<Program>(std::move(defs));
};

llvm::Expected<std::unique_ptr<ConstDef>> Parser::parseConstDef() {
  auto nameTok = expect_next(IDENTIFIER);
  if (!nameTok)
    return nameTok.takeError();
  std::string_view name = nameTok->getValue();

  auto colonTok = expect_next(COLON);
  if (!colonTok)
    return colonTok.takeError();

  next(); // consume ':'
  auto result = parseScalarType();
  if (!result)
    return result.takeError();
  auto type = std::move(*result);

  auto equalTok = expect_next(EQUAL);
  if (!equalTok)
    return equalTok.takeError();

  auto numTok = expect_next(NUMBER);
  if (!numTok)
    return numTok.takeError();
  int value = std::stoi(std::string(numTok->getValue()));

  auto semiTok = expect_next(SEMICOLON);
  if (!semiTok)
    return semiTok.takeError();

  next(); // consume ';'

  return std::make_unique<ConstDef>(std::string(name), value, std::move(type));
}

llvm::Expected<std::unique_ptr<FunctionDef>> Parser::parseFunctionDef() {
  auto nameTok = expect_next(IDENTIFIER);
  if (!nameTok)
    return nameTok.takeError();
  std::string_view name = nameTok->getValue(); // FuncDef uses string. TODO

  auto leftParenTok = expect_next(LEFT_PAREN);
  if (!leftParenTok)
    return leftParenTok.takeError();

  next(); // consume '('
  std::vector<Param> params;
  while (current.kind != RIGHT_PAREN) {
    auto param = parseParam();
    if (!param)
      return param.takeError();
    params.push_back(std::move(*param));
    next(); // consume param type
    if (current.kind == COMMA) {
      next();
    }
  }
  if (current.kind != RIGHT_PAREN)
    return makeError("expected ')' to close the parameter list");

  auto arrowRightTok = expect_next(ARROW_RIGHT);
  if (!arrowRightTok)
    return arrowRightTok.takeError();

  next(); // consume '->'
  auto returnTypeTok = parseType();
  if (!returnTypeTok)
    return returnTypeTok.takeError();

  auto body = parseBody();
  if (!body)
    return body.takeError();

  if (current.kind == RIGHT_CURLY_BRACKET)
    return makeError("expected return expression in function body");

  auto returnExpr = parseExpression();
  if (!returnExpr)
    return returnExpr.takeError();

  if (current.kind != RIGHT_CURLY_BRACKET)
    return makeError("expected '}' to close the function body");
  next(); // consume '}'

  return std::make_unique<FunctionDef>(std::string(name), std::move(params),
                                       std::move(*body), std::move(*returnExpr),
                                       std::move(*returnTypeTok));
};

llvm::Expected<Param> Parser::parseParam() {
  auto identExpr = parseIdentifierExpr();
  if (!identExpr)
    return identExpr.takeError();

  auto colonTok = expect_next(COLON);
  if (!colonTok)
    return colonTok.takeError();

  next(); // consume ':'
  auto type = parseType();
  if (!type)
    return type.takeError();

  return Param(std::move(**identExpr), std::move(*type));
}

llvm::Expected<std::unique_ptr<Type>> Parser::parseType() {
  switch (current.kind) {
  case TENSOR:
    return parseTensorType();
  case FLOAT_TYPE:
  case INTEGER_TYPE:
    return parseScalarType();
  default:
    return makeError("expected type, got '" +
                     std::string(tokenKindName(current.kind)) + "'");
  }
}

llvm::Expected<std::unique_ptr<TensorType>> Parser::parseTensorType() {
  auto leftAngleTok = expect_next(LEFT_ANGLE);
  if (!leftAngleTok)
    return leftAngleTok.takeError();

  std::vector<int64_t> shape;
  next(); // consume '<'
  while (current.kind != RIGHT_ANGLE) {
    if (current.kind != NUMBER)
      return makeError("expected number in tensor shape, got '" +
                       std::string(tokenKindName(current.kind)) + "'");
    shape.push_back(std::stoll(std::string(current.getValue())));
    next(); // consume number
    if (current.kind == COMMA) {
      next();
    }
  }
  return std::make_unique<TensorType>(shape);
}

llvm::Expected<std::unique_ptr<ScalarType>> Parser::parseScalarType() {
  if (current.kind != INTEGER_TYPE && current.kind != FLOAT_TYPE)
    return makeError("expected type (f32 or i32), got '" +
                     std::string(tokenKindName(current.kind)) + "'");

  ScalarKind kind =
      (current.kind == INTEGER_TYPE) ? ScalarKind::Integer : ScalarKind::Float;
  return std::make_unique<ScalarType>(kind);
}

llvm::Expected<std::vector<std::unique_ptr<Statement>>> Parser::parseBody() {
  auto leftCurlyBracket = expect_next(LEFT_CURLY_BRACKET);
  if (!leftCurlyBracket)
    return leftCurlyBracket.takeError();

  next(); // consume '{'
  std::vector<std::unique_ptr<Statement>> statements;
  while (current.kind != RIGHT_CURLY_BRACKET) {
    if (current.kind == LET) {
      auto letBinding = parseLetBinding();
      if (!letBinding)
        return letBinding.takeError();
      statements.push_back(std::move(*letBinding));
    } else {
      break;
    }
  }

  return statements;
}

llvm::Expected<std::unique_ptr<LetBinding>> Parser::parseLetBinding() {
  next(); // consume 'let'
  auto identExpr = parseIdentifierExpr();
  if (!identExpr)
    return identExpr.takeError();

  auto equalTok = expect_next(EQUAL);
  if (!equalTok)
    return equalTok.takeError();

  next();
  auto expr = parseExpression();
  if (!expr)
    return expr.takeError();

  if (current.kind != SEMICOLON)
    return makeError("expected ';', got '" +
                     std::string(tokenKindName(current.kind)) + "'");
  next(); // consume ';'

  return std::make_unique<LetBinding>(std::move(**identExpr), std::move(*expr));
}

llvm::Expected<std::unique_ptr<IdentifierExpr>> Parser::parseIdentifierExpr() {
  if (current.kind != IDENTIFIER)
    return makeError("expected 'identifier', got '" +
                     std::string(tokenKindName(current.kind)) + "'");
  return std::make_unique<IdentifierExpr>(std::string(current.getValue()));
}

llvm::Expected<std::unique_ptr<ScalarLiteralExpr>>
Parser::parseScalarLiteralExpr() {
  auto value = current.getValue();
  next();
  if (value.find('.') != std::string::npos)
    return std::make_unique<ScalarLiteralExpr>(std::stod(std::string(value)),
                                               ScalarKind::Float);
  else
    return std::make_unique<ScalarLiteralExpr>(std::stod(std::string(value)),
                                               ScalarKind::Integer);
}

llvm::Expected<std::unique_ptr<CallExpr>>
Parser::parseCallExpr(std::string name) {
  next(); // consume left paren
  std::vector<std::unique_ptr<Expression>> args;
  while (current.kind != RIGHT_PAREN) {
    auto arg = parseExpression();
    if (!arg)
      return arg.takeError();
    args.push_back(std::move(*arg));
    if (current.kind == COMMA)
      next();
  }
  next();
  return std::make_unique<CallExpr>(IdentifierExpr(name), std::move(args),
                                    getBuiltinOp(name));
}

llvm::Expected<std::unique_ptr<TensorExpr>> Parser::parseTensorExpr() {
  auto leftAngle = expect_next(LEFT_ANGLE);
  if (!leftAngle)
    return leftAngle.takeError();
  next(); // consume '<'

  std::vector<int64_t> shape;
  while (current.kind != RIGHT_ANGLE) {
    if (current.kind != NUMBER)
      return makeError("expected number in tensor shape, got '" +
                       std::string(tokenKindName(current.kind)) + "'");
    shape.push_back(std::stoll(std::string(current.getValue())));
    next(); // consume NUMBER
    if (current.kind == COMMA)
      next();
  }
  auto leftParen = expect_next(LEFT_PAREN);
  if (!leftParen)
    return leftParen.takeError();
  next(); // consume '('

  std::vector<std::unique_ptr<ScalarLiteralExpr>> values;
  while (current.kind != RIGHT_PAREN) {
    if (current.kind != NUMBER)
      return makeError("expected number in tensor values, got '" +
                       std::string(tokenKindName(current.kind)) + "'");
    auto value = current.getValue();
    if (value.find('.') != std::string::npos)
      values.push_back(std::make_unique<ScalarLiteralExpr>(
          std::stod(std::string(value)), ScalarKind::Float));
    else
      values.push_back(std::make_unique<ScalarLiteralExpr>(
          std::stod(std::string(value)), ScalarKind::Integer));
    next(); // consume NUMBER
    if (current.kind == COMMA)
      next();
  }
  next(); // consume ')'
  return std::make_unique<TensorExpr>(shape, std::move(values));
}

llvm::Expected<std::unique_ptr<PermutationExpr>>
Parser::parsePermutationExpr() {
  next(); // consume '['
  std::vector<size_t> values;
  while (current.kind != RIGHT_BRACKET) {
    if (current.kind != NUMBER)
      return makeError("expected integer in permutation, got '" +
                       std::string(tokenKindName(current.kind)) + "'");
    values.push_back(std::stoull(std::string(current.getValue())));
    next(); // consume number
    if (current.kind == COMMA)
      next();
  }
  next(); // consume ']'
  return std::make_unique<PermutationExpr>(std::move(values));
}

llvm::Expected<std::unique_ptr<Expression>> Parser::parseExpression() {
  switch (current.kind) {
  case TENSOR:
    return parseTensorExpr();
  case NUMBER:
    return parseScalarLiteralExpr();
  case LEFT_BRACKET:
    return parsePermutationExpr();
  case IDENTIFIER: {
    std::string name = std::string(current.getValue());
    next(); // consume identifier
    if (current.kind == LEFT_PAREN)
      return parseCallExpr(name);
    return std::make_unique<IdentifierExpr>(name);
  }
  default:
    return makeError("expected expression, got '" +
                     std::string(tokenKindName(current.kind)) + "'");
  }
}