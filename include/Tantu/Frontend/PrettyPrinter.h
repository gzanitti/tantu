#include "Tantu/Frontend/AST.h"
class PrettyPrinter : public Visitor {
public:
  virtual ~PrettyPrinter() = default;
  void visit(LetBinding &stmt) override;
  void visit(Param &param) override;
  void visit(IdentifierExpr &expr) override;
  void visit(PermutationExpr &expr) override;
  void visit(ScalarLiteralExpr &expr) override;
  void visit(TensorType &type) override;
  void visit(ScalarType &type) override;
  void visit(CallExpr &expr) override;
  void visit(TensorExpr &expr) override;
  void visit(FunctionDef &def) override;
  void visit(ConstDef &def) override;
  void visit(Program &prog) override;
};