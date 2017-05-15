// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:meta/meta.dart";

import "package:tensor_math/tensor_math.dart";

import "executable.dart";
import "operation.dart";

import "impl/tensor.dart";

export "impl/core.dart" show toNDArray, TensorBase;
export "impl/tensor.dart"
    show DefaultTensorBase, DefaultDifferentiableTensorBase;

abstract class Tensor implements Executable {
  // TODO implementare: DataType get dataType;

  NDShape get shape;

  void setShapeDimensions(List<int> newDimensions);

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
}

abstract class Constant implements Tensor {
  factory Constant(value, {String name}) =>
      new ConstantImpl(value, name: name);
}

abstract class ZerosLike implements Tensor {
  factory ZerosLike(input, {String name}) =>
      new ZerosLikeImpl(input, name: name);
}

abstract class OnesLike implements Tensor {
  factory OnesLike(input, {String name}) => new OnesLikeImpl(input, name: name);
}

abstract class Named implements Tensor {
  factory Named(target, {@required String name}) =>
      new NamedImpl(target, name: name);
}

abstract class Placeholder implements Tensor {
  factory Placeholder({String name, @required List<num> shapeDimensions}) =>
      new PlaceholderImpl(null, shapeDimensions: shapeDimensions, name: name);

  factory Placeholder.withDefault(defaultInput,
          {String name, List<num> shapeDimensions}) =>
      new PlaceholderImpl(defaultInput,
          shapeDimensions: shapeDimensions, name: name);
}

abstract class DefaultTensorDescriptor {
  bool get isCalculatingShape;

  NDShapeable toNDShapeable(value);

  Iterable<String> get inputNames;

  bool hasInput(String name);

  Tensor getInput(String name);

  NDShapeable getInputValue(String name);
}

abstract class OutputGradientComputersDescriptor {
  Iterable<String> get inputNames;

  bool hasInput(String name);

  Tensor getInput(String name);

  void setOutputGradient(
      String inputName, TensorGradientComputer gradientComputer);
}
