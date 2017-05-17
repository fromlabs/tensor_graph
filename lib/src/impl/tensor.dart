// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:meta/meta.dart";

import "package:tensor_math/tensor_math.dart";

import "../operation.dart";
import "../tensor.dart";

abstract class DefaultTensorBase extends TensorBase {
  DefaultTensorBase(
      Map<String, dynamic> inputs, String operationName, String type)
      : super() {
    new _DefaultOutputOperationImpl(this, inputs, operationName, type);
  }

  DefaultTensorBase.output(Map<String, dynamic> inputs, String operationName,
      String outputName, String type)
      : super() {
    new _DefaultOutputOperationImpl.output(
        this, inputs, operationName, outputName, type);
  }

  @protected
  NDShapeable computeValue(DefaultTensorDescriptor descriptor);
}

abstract class DefaultDifferentiableTensorBase extends DefaultTensorBase {
  DefaultDifferentiableTensorBase(
      Map<String, dynamic> inputs, String operationName, String type)
      : super(inputs, operationName, type);

  DefaultDifferentiableTensorBase.output(Map<String, dynamic> inputs,
      String operationName, String outputName, String type)
      : super.output(inputs, operationName, outputName, type);

  @protected
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor);
}

class ConstantImpl extends DefaultTensorBase implements Constant {
  static const String __type = "Constant";

  final NDArray _value;

  factory ConstantImpl(value, {String name}) =>
      new ConstantImpl._(toNDArray(value), name);

  ConstantImpl._(this._value, String name) : super(null, name, __type);

  @override
  NDShapeable computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.toNDShapeable(_value);
}

class ZerosLikeImpl extends DefaultTensorBase implements ZerosLike {
  static const String __type = "ZerosLike";

  static const String _inputName = "input";

  ZerosLikeImpl(_input, {String name})
      : super({_inputName: _input}, name, __type);

  @override
  NDShapeable computeValue(DefaultTensorDescriptor descriptor) {
    var inputValue = descriptor.getInputValue(_inputName);

    if (inputValue is NDArray) {
      return new NDArray(new List.filled(inputValue.shape.length, 0))
          .reshape(newDimensions: inputValue.shape.dimensions);
    } else {
      return inputValue;
    }
  }
}

class OnesLikeImpl extends DefaultTensorBase implements OnesLike {
  static const String __type = "OnesLike";

  static const String _inputName = "input";

  OnesLikeImpl(_input, {String name})
      : super({_inputName: _input}, name, __type);

  @override
  NDShapeable computeValue(DefaultTensorDescriptor descriptor) {
    var inputValue = descriptor.getInputValue(_inputName);

    if (inputValue is NDArray) {
      return new NDArray(new List.filled(inputValue.shape.length, 1))
          .reshape(newDimensions: inputValue.shape.dimensions);
    } else {
      return inputValue;
    }
  }
}

class NamedImpl extends DefaultDifferentiableTensorBase implements Reference {
  static const String __type = "Named";

  static const String _targetInputName = "target";

  NamedImpl(target, {String name})
      : super({_targetInputName: target}, name, __type) {
    if (name == null) {
      throw new ArgumentError("Named $this must specify a name");
    }
  }

  @override
  NDShapeable computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_targetInputName);

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _targetInputName,
        (TensorGradientDescriptor descriptor) =>
            descriptor.backPropagatedGradientValue);
  }
}

class ModelInputImpl extends DefaultDifferentiableTensorBase
    implements ModelInput {
  static const String __type = "ModelInput";

  static const String _defaultInputName = "default";

  ModelInputImpl(target, {String name, List<num> shapeDimensions})
      : super({_defaultInputName: target}, name, __type) {
    if (target == null && shapeDimensions == null) {
      throw new ArgumentError(
          "ModelInput $this must specify at least a default value or a shape");
    }

    setShapeDimensions(shapeDimensions);
  }

  @override
  NDShapeable computeValue(DefaultTensorDescriptor descriptor) {
    if (!descriptor.isCalculatingShape) {
      if (descriptor.hasInput(_defaultInputName)) {
        return descriptor.getInputValue(_defaultInputName);
      } else {
        throw new StateError(
            "Placeholder $this without a default value should be feeded");
      }
    } else {
      if (descriptor.hasInput(_defaultInputName)) {
        return descriptor.getInputValue(_defaultInputName);
      } else {
        return new NDShape();
      }
    }
  }

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _defaultInputName,
        (TensorGradientDescriptor descriptor) =>
            descriptor.backPropagatedGradientValue);
  }
}

class _DefaultOutputOperationImpl extends OperationBase {
  _DefaultOutputOperationImpl(Tensor defaultOutput, Map<String, dynamic> inputs,
      String operationName, String type)
      : super(inputs, operationName, type) {
    registerDefaultOutputProduced(defaultOutput);
  }

  _DefaultOutputOperationImpl.output(Tensor output, Map<String, dynamic> inputs,
      String operationName, String outputName, String type)
      : super(inputs, operationName, type) {
    registerOutputProduced(outputName, output);
  }

  Tensor get singleOutput => getOutput(outputNames.first);

  @override
  void computeOperation(OperationDescriptor descriptor) {
    DefaultTensorBase output = singleOutput;
    descriptor.setOutputValue(singleOutput.operationOutputName,
        output.computeValue(new _DefaultTensorDescriptorImpl(descriptor)));
  }

  @override
  void buildGradients(GradientsComputersDescriptor descriptor) {
    var defaultDescriptor = new _DefaultGradientsComputersDescriptorImpl(
        singleOutput.operationOutputName, descriptor);

    if (singleOutput is DefaultDifferentiableTensorBase) {
      DefaultDifferentiableTensorBase output = singleOutput;

      output.buildDefaultGradients(defaultDescriptor);
    }
  }
}

class _DefaultTensorDescriptorImpl implements DefaultTensorDescriptor {
  final OperationDescriptor _descriptor;

  _DefaultTensorDescriptorImpl(this._descriptor);

  @override
  bool get isCalculatingShape => _descriptor.isCalculatingShape;

  @override
  NDShapeable toNDShapeable(value) =>
      isCalculatingShape ? toNDArray(value).shape : toNDArray(value);

  @override
  Iterable<String> get inputNames => _descriptor.inputNames;

  @override
  bool hasInput(String name) => _descriptor.hasInput(name);

  @override
  Tensor getInput(String name) => _descriptor.getInput(name);

  @override
  NDShapeable getInputValue(String name) => _descriptor.getInputValue(name);
}

class _DefaultGradientsComputersDescriptorImpl
    implements OutputGradientComputersDescriptor {
  final String _outputName;

  final GradientsComputersDescriptor _descriptor;

  _DefaultGradientsComputersDescriptorImpl(this._outputName, this._descriptor);

  @override
  Iterable<String> get inputNames => _descriptor.inputNames;

  @override
  bool hasInput(String name) => _descriptor.hasInput(name);

  @override
  Tensor getInput(String name) => _descriptor.getInput(name);

  @override
  void setOutputGradient(
      String inputName, TensorGradientComputer gradientComputer) {
    _descriptor.setOutputGradient(_outputName, inputName, gradientComputer);
  }
}
