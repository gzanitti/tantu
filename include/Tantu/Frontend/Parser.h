

#include "AST.h"
#include "Lexer.h"
#include <llvm/Support/raw_ostream.h>
#include <memory>
#include <vector>

class Parser {

public:
  Parser(Lexer lexer);
  llvm::Expected<std::unique_ptr<Program>> parseProgram();

private:
  Lexer lexer;
  Token current;

  llvm::Expected<std::unique_ptr<ConstDef>> parseConstDef();
  llvm::Expected<std::unique_ptr<FunctionDef>> parseFunctionDef();
  llvm::Expected<std::unique_ptr<Expression>> parseExpression();
  llvm::Expected<std::unique_ptr<TensorExpr>> parseTensorExpr();
  llvm::Expected<std::unique_ptr<CallExpr>> parseCallExpr(std::string name);
  llvm::Expected<std::unique_ptr<IdentifierExpr>> parseIdentifierExpr();
  llvm::Expected<std::unique_ptr<LetBinding>> parseLetBinding();
  llvm::Expected<Param> parseParam();
  llvm::Expected<std::unique_ptr<ScalarLiteralExpr>> parseScalarLiteralExpr();
  llvm::Expected<std::unique_ptr<PermutationExpr>> parsePermutationExpr();
  llvm::Expected<std::unique_ptr<Type>> parseType();
  llvm::Expected<std::unique_ptr<TensorType>> parseTensorType();
  llvm::Expected<std::unique_ptr<ScalarType>> parseScalarType();
  llvm::Expected<std::vector<std::unique_ptr<Statement>>> parseBody();

  Token next();

  llvm::Error makeError(std::string message);
  llvm::Expected<Token> expect_next(TokenKind kind);
};
