// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:tensor_math/tensor_math.dart";

import "../operation.dart";
import "../tensor.dart";
import "../variable.dart";

class VariableImpl extends DefaultTensorBase implements Variable {
  static const String __type = "Variable";

  static const String _initialValueInputName = "initial_value";

  static const String _initializerSingletonKey = "_INITIALIZER";

  static const String _variableStateKey = "_VARIABLE";

  VariableImpl(initialValue, {NDDataType dataType, String name})
      : super(
            inputs: {_initialValueInputName: initialValue},
            operationName: name,
            type: __type,
            dataType: dataType);

  @override
  Tensor get initialValue => getInput(_initialValueInputName);

  @override
  NDObject computeValue(DefaultTensorDescriptor descriptor) {
    if (!descriptor.isEvaluatingDescriptor) {
      return state.getFromSession(_variableStateKey) ??
          (throw new StateError("Variable $this uninitialized"));
    } else {
      return descriptor.getInputValue(_initialValueInputName);
    }
  }

  @override
  Operation get initializer => provideGraphContextualizedSingleton(
      _initializerSingletonKey,
      () => new _VariableAssign(this, this.initialValue).operation);

  @override
  Tensor assign(value) => new _VariableAssign(this, value);

  NDArray _assignEvaluation(NDArray evaluation) {
    state.setInSession(_variableStateKey, evaluation);
    return evaluation;
  }
}

class _VariableAssign extends DefaultTensorBase {
  static const String __type = "Assign";

  static const String _valueInputName = "value";

  final VariableImpl _variable;

  _VariableAssign(this._variable, value)
      : super(
            inputs: {_valueInputName: value},
            operationName: "${_variable.operation.id}/$__type",
            type: __type);

  @override
  NDObject computeValue(DefaultTensorDescriptor descriptor) {
    var inputValue = descriptor.getInputValue(_valueInputName);
    if (inputValue is NDArray) {
      return _variable._assignEvaluation(inputValue);
    } else {
      return inputValue;
    }
  }
}
