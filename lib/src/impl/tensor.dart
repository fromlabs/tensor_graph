// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:meta/meta.dart";

import "package:tensor_math/tensor_math.dart";

import "../operation.dart";
import "../tensor.dart";

abstract class DefaultTensorBase extends TensorBase {
  DefaultTensorBase(
      {@required String type,
      Map<String, dynamic> inputs,
      String operationName,
      NDDataType dataType})
      : super(dataType: dataType) {
    new _DefaultOutputOperationImpl(this, inputs, operationName, type);
  }

  DefaultTensorBase.output(
      {@required String type,
      Map<String, dynamic> inputs,
      String operationName,
      String outputName,
      NDDataType dataType})
      : super(dataType: dataType) {
    new _DefaultOutputOperationImpl.output(
        this, inputs, operationName, outputName, type);
  }

  @protected
  NDObject computeValue(DefaultTensorDescriptor descriptor);
}

abstract class DefaultDifferentiableTensorBase extends DefaultTensorBase {
  DefaultDifferentiableTensorBase(
      {@required String type,
      Map<String, dynamic> inputs,
      String operationName,
      NDDataType dataType})
      : super(
            type: type,
            inputs: inputs,
            operationName: operationName,
            dataType: dataType);

  DefaultDifferentiableTensorBase.output(Map<String, dynamic> inputs,
      String operationName, String outputName, String type, NDDataType dataType)
      : super.output(
            type: type,
            inputs: inputs,
            operationName: operationName,
            outputName: outputName,
            dataType: dataType);

  @protected
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor);
}

class ConstantImpl extends DefaultTensorBase implements Constant {
  static const String __type = "Constant";

  final NDArray _value;

  factory ConstantImpl(value, {NDDataType dataType, String name}) =>
      new ConstantImpl._(toNDArray(value, dataType: dataType), name, dataType);

  ConstantImpl._(this._value, String name, NDDataType dataType)
      : super(operationName: name, type: __type, dataType: dataType);

  @override
  NDObject computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.toNDObject(_value);
}

class ZerosLikeImpl extends DefaultTensorBase implements ZerosLike {
  static const String __type = "ZerosLike";

  static const String _inputName = "input";

  ZerosLikeImpl(_input, {NDDataType dataType, String name})
      : super(
            inputs: {_inputName: _input},
            operationName: name,
            type: __type,
            dataType: dataType);

  @override
  NDObject computeValue(DefaultTensorDescriptor descriptor) {
    var inputValue = descriptor.getInputValue(_inputName);

    if (descriptor.isEvaluatingDescriptor) {
      return inputValue;
    } else {
      // TODO gestione valida per tutti i numeri
      return new NDArray(new List.filled(inputValue.shape.length, 0.0),
              dataType: dataType)
          .reshape(newDimensions: inputValue.shape.dimensions);
    }
  }
}

class OnesLikeImpl extends DefaultTensorBase implements OnesLike {
  static const String __type = "OnesLike";

  static const String _inputName = "input";

  OnesLikeImpl(_input, {NDDataType dataType, String name})
      : super(
            inputs: {_inputName: _input},
            operationName: name,
            type: __type,
            dataType: dataType);

  @override
  NDObject computeValue(DefaultTensorDescriptor descriptor) {
    var inputValue = descriptor.getInputValue(_inputName);

    if (descriptor.isEvaluatingDescriptor) {
      return inputValue;
    } else {
      // TODO gestione valida per tutti i numeri
      return new NDArray(new List.filled(inputValue.shape.length, 1.0),
              dataType: dataType)
          .reshape(newDimensions: inputValue.shape.dimensions);
    }
  }
}

class ReferenceImpl extends DefaultDifferentiableTensorBase
    implements Reference {
  static const String __type = "Reference";

  static const String _targetInputName = "target";

  ReferenceImpl(target, {NDDataType dataType, String name})
      : super(
            inputs: {_targetInputName: target},
            operationName: name,
            type: __type,
            dataType: dataType) {
    if (name == null) {
      throw new ArgumentError("Reference $this must specify a name");
    }
  }

  @override
  NDObject computeValue(DefaultTensorDescriptor descriptor) =>
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

  ModelInputImpl(target,
      {List<int> shapeDimensions, NDDataType dataType, String name})
      : super(
            inputs: {_defaultInputName: target},
            operationName: name,
            type: __type,
            dataType: dataType) {
    if (target == null && (shapeDimensions == null || dataType == null)) {
      throw new ArgumentError(
          "ModelInput $this must specify at least a default value or a shape and a data type");
    }

    setShapeDimensions(shapeDimensions);
  }

  @override
  NDObject computeValue(DefaultTensorDescriptor descriptor) {
    if (!descriptor.isEvaluatingDescriptor) {
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
        return new NDDescriptor();
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
      : super(inputs: inputs, name: operationName, type: type) {
    registerDefaultOutputProduced(defaultOutput);
  }

  _DefaultOutputOperationImpl.output(Tensor output, Map<String, dynamic> inputs,
      String operationName, String outputName, String type)
      : super(inputs: inputs, name: operationName, type: type) {
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
  bool get isEvaluatingDescriptor => _descriptor.isEvaluatingDescriptor;

  @override
  NDObject toNDObject(value, {@required NDDataType dataType}) {
    var array = toNDArray(value, dataType: dataType);
    return isEvaluatingDescriptor ? array.descriptor : array;
  }

  @override
  Iterable<String> get inputNames => _descriptor.inputNames;

  @override
  bool hasInput(String name) => _descriptor.hasInput(name);

  @override
  Tensor getInput(String name) => _descriptor.getInput(name);

  @override
  NDObject getInputValue(String name) => _descriptor.getInputValue(name);
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
