// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "../operation.dart";
import "../tensor.dart";
import "../variable.dart";

class VariableImpl extends DefaultTensorBase implements Variable {
  static const String __type = "Variable";

  static const String _initialValueInputName = "initial_value";

  static const String _initializerSingletonKey = "_INITIALIZER";

  static const String _variableStateKey = "_VARIABLE";

  VariableImpl(initialValue, {String name})
      : super({_initialValueInputName: initialValue}, name, __type);

  @override
  Tensor get initialValue => getInput(_initialValueInputName);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      state.getFromSession(_variableStateKey) ??
      (throw new StateError("Variable $this uninitialized"));

  @override
  Operation get initializer => provideGraphContextualizedSingleton(
      _initializerSingletonKey,
      () => new _VariableAssign(this, this.initialValue).operation);

  @override
  Tensor assign(value) => new _VariableAssign(this, value);

  dynamic _assignEvaluation(evaluation) {
    state.setInSession(_variableStateKey, evaluation);
    return evaluation;
  }
}

class _VariableAssign extends DefaultTensorBase {
  static const String __type = "Assign";

  static const String _valueInputName = "value";

  final VariableImpl _variable;

  _VariableAssign(this._variable, value)
      : super({_valueInputName: value}, "${_variable.operation.id}/$__type",
            __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      _variable._assignEvaluation(descriptor.getInputValue(_valueInputName));
}
