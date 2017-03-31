// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:meta/meta.dart";

import "package:tensor_math/tensor_math.dart" as math;

import "../operation.dart";
import "../tensor.dart";

dynamic toValue(dynamic value) {
  // TODO check del valore

  // TODO eventuale conversione in una struttura immutabile

  return value;
}

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
  dynamic computeValue(DefaultTensorDescriptor descriptor);
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

  final dynamic _value;

  ConstantImpl(_value, {String name})
      : this._value = toValue(_value),
        super(null, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => _value;
}

class ReferenceImpl extends DefaultDifferentiableTensorBase
    implements Reference {
  static const String __type = "Reference";

  static const String _targetInputName = "target";

  ReferenceImpl({target, String name})
      : super({_targetInputName: target}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) {
    if (!descriptor.hasInput(_targetInputName)) {
      throw new StateError(
          "Reference $this without a target should be feeded");
    }

    return descriptor.getInputValue(_targetInputName);
  }

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _targetInputName,
        (TensorGradientDescriptor descriptor) =>
            math.mul(1, descriptor.backPropagatedGradientValue));
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
  Iterable<String> get inputNames => _descriptor.inputNames;

  @override
  bool hasInput(String name) => _descriptor.hasInput(name);

  @override
  Tensor getInput(String name) => _descriptor.getInput(name);

  @override
  dynamic getInputValue(String name) => _descriptor.getInputValue(name);
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
