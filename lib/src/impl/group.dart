// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:meta/meta.dart";

import "../executable.dart";
import "../operation.dart";
import "../tensor.dart";
import "../group.dart";

class GroupOperationImpl extends GroupOperationBase implements GroupOperation {
  static const String __type = "Group";

  final GroupBuilder _builder;

  GroupOperationImpl(Map<String, dynamic> inputs, this._builder, {String name})
      : super(inputs, name, __type);

  @override
  void buildOperation(GroupDescriptor descriptor) {
    _builder(descriptor);
  }
}

class DefaultGroupTensorImpl extends DefaultGroupTensorBase
    implements DefaultGroupTensor {
  static const String __type = "DefaultGroupTensor";

  final DefaultGroupTensorBuilder _builder;

  DefaultGroupTensorImpl(Map<String, dynamic> inputs, this._builder,
      {String name})
      : super(inputs, name, __type);

  @override
  Tensor buildValue(DefaultGroupTensorDescriptor descriptor) =>
      _builder(descriptor);
}

abstract class DefaultGroupTensorBase extends TensorBase {
  DefaultGroupTensorBase(
      Map<String, dynamic> inputs, String operationName, String type)
      : super(null) {
    new _DefaultGroupOperationImpl(inputs, this, operationName, type);
  }

  DefaultGroupTensorBase.output(Map<String, dynamic> inputs,
      String operationName, String outputName, String type)
      : super(null) {
    new _DefaultGroupOperationImpl.output(
        inputs, this, operationName, outputName, type);
  }

  @protected
  Tensor buildValue(DefaultGroupTensorDescriptor descriptor);
}

class _DefaultGroupTensorDescriptorImpl
    implements DefaultGroupTensorDescriptor {
  final _DefaultGroupOperationImpl _group;

  final Map<String, Tensor> _internalInputs;

  _DefaultGroupTensorDescriptorImpl(this._group, this._internalInputs);

  @override
  Iterable<String> get inputNames => _internalInputs.keys;

  @override
  bool hasInput(String name) => _internalInputs.containsKey(name);

  @override
  Tensor getInput(String name) =>
      _internalInputs[name] ??
      (throw new ArgumentError.value(
          name, "Input not specified in $_group descriptor"));

  @override
  bool hasImport(Executable executable) => _group.hasImport(executable);

  @override
  E import<E extends Executable>(E executable) => _group.import(executable);
}

class _DefaultGroupOperationImpl extends GroupOperationBase {
  final DefaultGroupTensorBase _singleOutput;

  final String _outputName;

  _DefaultGroupOperationImpl(Map<String, dynamic> inputs, this._singleOutput,
      String operationName, String type)
      : this._outputName = Operation.defaultOutputName,
        super(inputs, operationName, type);

  _DefaultGroupOperationImpl.output(Map<String, dynamic> inputs,
      this._singleOutput, String operationName, this._outputName, String type)
      : super(inputs, operationName, type);

  @override
  void buildOperation(GroupDescriptor descriptor) {
    var inputs = new Map<String, Tensor>.fromIterable(descriptor.inputNames,
        value: (inputName) => descriptor.getInput(inputName));

    var defaultDescriptor = new _DefaultGroupTensorDescriptorImpl(this, inputs);

    var internalDefaultOutput = _singleOutput.buildValue(defaultDescriptor);

    if (internalDefaultOutput != null) {
      descriptor.setOutput(_outputName, internalDefaultOutput);
    } else {
      throw new ArgumentError.notNull("Internal default output");
    }
  }

  @override
  @protected
  Tensor createExternalOutput(String outputName) =>
      outputName == Operation.defaultOutputName
          ? _singleOutput
          : super.createExternalOutput(outputName);
}
