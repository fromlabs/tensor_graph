// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:meta/meta.dart";

import "package:tensor_math/tensor_math.dart";

import "executable.dart";
import "operation.dart";

import "impl/tensor.dart";

export "impl/core.dart" show TensorBase;
export "impl/tensor.dart"
    show DefaultTensorBase, DefaultDifferentiableTensorBase;

abstract class Tensor implements Executable {
  NDDescriptor get descriptor;

  NDDataType get dataType;

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
  factory Constant(value,
          {NDDataType dataType = NDDataType.float32, String name}) =>
      new ConstantImpl(value, dataType: dataType, name: name);
}

abstract class ZerosLike implements Tensor {
  factory ZerosLike(input,
          {NDDataType dataType = NDDataType.float32, String name}) =>
      new ZerosLikeImpl(input, dataType: dataType, name: name);
}

abstract class OnesLike implements Tensor {
  factory OnesLike(input,
          {NDDataType dataType = NDDataType.float32, String name}) =>
      new OnesLikeImpl(input, dataType: dataType, name: name);
}

abstract class Reference implements Tensor {
  factory Reference(target, {@required String name}) =>
      new ReferenceImpl(target, name: name);
}

abstract class ModelInput implements Tensor {
  factory ModelInput(
          {@required List<int> shapeDimensions,
          NDDataType dataType = NDDataType.float32,
          String name}) =>
      new ModelInputImpl(null,
          shapeDimensions: shapeDimensions, dataType: dataType, name: name);

  factory ModelInput.withDefault(defaultInput,
          {List<int> shapeDimensions, NDDataType dataType, String name}) =>
      new ModelInputImpl(defaultInput,
          shapeDimensions: shapeDimensions, name: name);
}

abstract class DefaultTensorDescriptor {
  bool get isEvaluatingDescriptor;

  NDObject toNDObject(value, {NDDataType dataType});

  Iterable<String> get inputNames;

  bool hasInput(String name);

  Tensor getInput(String name);

  NDObject getInputValue(String name);
}

abstract class OutputGradientComputersDescriptor {
  Iterable<String> get inputNames;

  bool hasInput(String name);

  Tensor getInput(String name);

  void setOutputGradient(
      String inputName, TensorGradientComputer gradientComputer);
}
