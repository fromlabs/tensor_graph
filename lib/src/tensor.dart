// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "executable.dart";
import "operation.dart";

import "impl/tensor.dart";

export "impl/core.dart" show TensorBase;
export "impl/tensor.dart"
    show DefaultTensorBase, DefaultDifferentiableTensorBase;

abstract class Tensor implements Executable {
  Operation get operation;

  bool get isDefaultOutput;

  String get operationOutputName;

  Iterable<String> get consumerIds;

  bool isDifferentiable(String inputName);

  bool get isEvaluated;

  bool get isNotEvaluated;

  bool get isExecutionValue;

  bool get isFeedValue;

  Tensor operator +(value);

  Tensor operator -(value);

  Tensor operator -();

  Tensor operator *(value);

  Tensor operator /(value);

  Tensor operator >(value);

  Tensor operator >=(value);

  Tensor operator <(value);

  Tensor operator <=(value);

// TODO implementare: DimensionType get dimensionType;
// TODO implementare: TensorShape shape;
}

abstract class Constant implements Tensor {
  factory Constant(value, {String name}) => new ConstantImpl(value, name: name);
}

abstract class Reference implements Tensor {
  factory Reference({target, String name}) =>
      new ReferenceImpl(target: target, name: name);
}

abstract class DefaultTensorDescriptor {
  Iterable<String> get inputNames;

  bool hasInput(String name);

  Tensor getInput(String name);

  dynamic getInputValue(String name);
}

abstract class OutputGradientComputersDescriptor {
  Iterable<String> get inputNames;

  bool hasInput(String name);

  Tensor getInput(String name);

  void setOutputGradient(
      String inputName, TensorGradientComputer gradientComputer);
}
