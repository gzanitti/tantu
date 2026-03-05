include "mlir/IR/OpBase.td"

def Tantu_Dialect : Dialect {
  let name = "tantu";
  let summary = "A minimal functional array language for numerical computation";
  let cppNamespace = "::tantu";
}