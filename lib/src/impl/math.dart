// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:collection/collection.dart";
import "package:tensor_math/tensor_math.dart" as math;

import "../operation.dart";
import "../tensor.dart";
import "../group.dart";
import "../math.dart";

class AddsImpl extends DefaultDifferentiableTensorBase implements Adds {
  static const String __type = "Adds";

  static const String _inputsInputName = "inputs";

  AddsImpl(List inputs, {String name})
      : super(
            mapMap(inputs.asMap(),
                key: (index, _) => "$_inputsInputName${index + 1}"),
            name,
            __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      math.adds(descriptor.inputNames
          .map((inputName) => descriptor.getInputValue(inputName))
          .toList());

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    for (var inputName in descriptor.inputNames) {
      descriptor.setOutputGradient(
          inputName,
          (TensorGradientDescriptor descriptor) =>
              math.mul(1, descriptor.backPropagatedGradientValue));
    }
  }
}

class AddImpl extends DefaultDifferentiableTensorBase implements Add {
  static const String __type = "Add";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  AddImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => math.add(
      descriptor.getInputValue(_input1InputName),
      descriptor.getInputValue(_input2InputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _input1InputName,
        (TensorGradientDescriptor descriptor) =>
            math.mul(1, descriptor.backPropagatedGradientValue));

    descriptor.setOutputGradient(
        _input2InputName,
        (TensorGradientDescriptor descriptor) =>
            math.mul(1, descriptor.backPropagatedGradientValue));
  }
}

class SubImpl extends DefaultDifferentiableTensorBase implements Sub {
  static const String __type = "Sub";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  SubImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => math.sub(
      descriptor.getInputValue(_input1InputName),
      descriptor.getInputValue(_input2InputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _input1InputName,
        (TensorGradientDescriptor descriptor) =>
            math.mul(1, descriptor.backPropagatedGradientValue));

    descriptor.setOutputGradient(
        _input2InputName,
        (TensorGradientDescriptor descriptor) =>
            math.mul(-1, descriptor.backPropagatedGradientValue));
  }
}

class NegImpl extends DefaultDifferentiableTensorBase implements Neg {
  static const String __type = "Neg";

  static const String _inputInputName = "input";

  NegImpl(input, {String name}) : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      math.neg(descriptor.getInputValue(_inputInputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) =>
            math.mul(-1, descriptor.backPropagatedGradientValue));
  }
}

class MulImpl extends DefaultDifferentiableTensorBase implements Mul {
  static const String __type = "Mul";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  MulImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => math.mul(
      descriptor.getInputValue(_input1InputName),
      descriptor.getInputValue(_input2InputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _input1InputName,
        (TensorGradientDescriptor descriptor) => math.mul(
            descriptor.getInputValue(_input2InputName),
            descriptor.backPropagatedGradientValue));

    descriptor.setOutputGradient(
        _input2InputName,
        (TensorGradientDescriptor descriptor) => math.mul(
            descriptor.getInputValue(_input1InputName),
            descriptor.backPropagatedGradientValue));
  }
}

class DivImpl extends DefaultDifferentiableTensorBase implements Div {
  static const String __type = "Div";

  static const String _numeratorInputName = "numerator";
  static const String _denominatorInputName = "denominator";

  DivImpl(numerator, denominator, {String name})
      : super({
          _numeratorInputName: numerator,
          _denominatorInputName: denominator
        }, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => math.div(
      descriptor.getInputValue(_numeratorInputName),
      descriptor.getInputValue(_denominatorInputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _numeratorInputName,
        (TensorGradientDescriptor descriptor) => math.mul(
            math.inv(descriptor.getInputValue(_denominatorInputName)),
            descriptor.backPropagatedGradientValue));

    descriptor.setOutputGradient(
        _denominatorInputName,
        (TensorGradientDescriptor descriptor) => math.mul(
            -descriptor.getInputValue(_numeratorInputName) /
                (descriptor.getInputValue(_denominatorInputName) *
                    descriptor.getInputValue(_denominatorInputName)),
            descriptor.backPropagatedGradientValue));
  }
}

class InvImpl extends DefaultDifferentiableTensorBase implements Inv {
  static const String __type = "Inv";

  static const String _inputInputName = "input";

  InvImpl(input, {String name}) : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      math.inv(descriptor.getInputValue(_inputInputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) => math.mul(
            -math.inv(descriptor.getInputValue(_inputInputName) *
                descriptor.getInputValue(_inputInputName)),
            descriptor.backPropagatedGradientValue));
  }
}

class ExpImpl extends DefaultDifferentiableTensorBase implements Exp {
  static const String __type = "Exp";

  static const String _inputInputName = "input";

  ExpImpl(input, {String name}) : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      math.exp(descriptor.getInputValue(_inputInputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) => math.mul(
            math.exp(descriptor.getInputValue(_inputInputName)),
            descriptor.backPropagatedGradientValue));
  }
}

class LogImpl extends DefaultDifferentiableTensorBase implements Log {
  static const String __type = "Log";

  static const String _inputInputName = "input";

  LogImpl(input, {String name}) : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      math.log(descriptor.getInputValue(_inputInputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) => math.mul(
            math.inv(descriptor.getInputValue(_inputInputName)),
            descriptor.backPropagatedGradientValue));
  }
}

class AbsImpl extends DefaultDifferentiableTensorBase implements Abs {
  static const String __type = "Abs";

  static const String _inputInputName = "input";

  AbsImpl(input, {String name}) : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      math.abs(descriptor.getInputValue(_inputInputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) => math.mul(
            math.select(descriptor.getInputValue(_inputInputName) >= 0, 1, -1),
            descriptor.backPropagatedGradientValue));
  }
}

class SigmoidImpl extends DefaultDifferentiableTensorBase implements Sigmoid {
  static const String __type = "Sigmoid";

  static const String _inputInputName = "input";

  SigmoidImpl(input, {String name})
      : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => math.inv(math.add(
      1, math.exp(math.neg(descriptor.getInputValue(_inputInputName)))));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) => math.mul(
            descriptor.outputValue * (-descriptor.outputValue + 1),
            descriptor.backPropagatedGradientValue));
  }
}

class TanhImpl extends DefaultDifferentiableTensorBase implements Tanh {
  static const String __type = "Tanh";

  static const String _inputInputName = "input";

  TanhImpl(input, {String name})
      : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) {
    var e2x = math.exp(math.mul(2, descriptor.getInputValue(_inputInputName)));

    return math.div(math.sub(e2x, 1), math.add(e2x, 1));
  }

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) => math.mul(
            -(descriptor.getInputValue(_inputInputName) *
                    descriptor.getInputValue(_inputInputName)) +
                1,
            descriptor.backPropagatedGradientValue));
  }
}

class EqualImpl extends DefaultTensorBase implements Equal {
  static const String __type = "Equal";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  EqualImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => math.equal(
      descriptor.getInputValue(_input1InputName),
      descriptor.getInputValue(_input2InputName));
}

class NotEqualImpl extends DefaultTensorBase implements NotEqual {
  static const String __type = "NotEqual";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  NotEqualImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => math.notEqual(
      descriptor.getInputValue(_input1InputName),
      descriptor.getInputValue(_input2InputName));
}

class LessImpl extends DefaultTensorBase implements Less {
  static const String __type = "Less";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  LessImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => math.less(
      descriptor.getInputValue(_input1InputName),
      descriptor.getInputValue(_input2InputName));
}

class LessEqualImpl extends DefaultTensorBase implements LessEqual {
  static const String __type = "LessEqual";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  LessEqualImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => math.lessEqual(
      descriptor.getInputValue(_input1InputName),
      descriptor.getInputValue(_input2InputName));
}

class GreaterImpl extends DefaultTensorBase implements Greater {
  static const String __type = "Greater";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  GreaterImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => math.greater(
      descriptor.getInputValue(_input1InputName),
      descriptor.getInputValue(_input2InputName));
}

class GreaterEqualImpl extends DefaultTensorBase implements GreaterEqual {
  static const String __type = "GreaterEqual";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  GreaterEqualImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => math.greaterEqual(
      descriptor.getInputValue(_input1InputName),
      descriptor.getInputValue(_input2InputName));
}

class ReluImpl extends DefaultDifferentiableTensorBase implements Relu {
  static const String __type = "Relu";

  static const String _inputInputName = "input";

  ReluImpl(input, {String name})
      : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => math.select(
      descriptor.getInputValue(_inputInputName) >= 0,
      descriptor.getInputValue(_inputInputName),
      0);

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) => math.mul(
            math.select(descriptor.getInputValue(_inputInputName) >= 0, 1, 0),
            descriptor.backPropagatedGradientValue));
  }
}

class SelectImpl extends DefaultDifferentiableTensorBase implements Select {
  static const String __type = "Select";

  static const String _conditionInputInputName = "conditionInput";
  static const String _thenInputInputName = "thenInput";
  static const String _elseInputInputName = "elseInput";

  SelectImpl(conditionInput, thenInput, elseInput, {String name})
      : super({
          _conditionInputInputName: conditionInput,
          _thenInputInputName: thenInput,
          _elseInputInputName: elseInput
        }, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => math.select(
      descriptor.getInputValue(_conditionInputInputName),
      descriptor.getInputValue(_thenInputInputName),
      descriptor.getInputValue(_elseInputInputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _thenInputInputName,
        (TensorGradientDescriptor descriptor) => math.mul(
            math.select(
                descriptor.getInputValue(_conditionInputInputName), 1, 0),
            descriptor.backPropagatedGradientValue));

    descriptor.setOutputGradient(
        _elseInputInputName,
        (TensorGradientDescriptor descriptor) => math.mul(
            math.select(
                descriptor.getInputValue(_conditionInputInputName), 0, 1),
            descriptor.backPropagatedGradientValue));
  }
}

class Loss2Impl extends DefaultGroupTensorBase implements Loss2 {
  static const String __type = "Loss2";

  static const String _expectedInputName = "expected";
  static const String _logitInputName = "logit";

  Loss2Impl(expected, logit, {String name})
      : super({_expectedInputName: expected, _logitInputName: logit}, name,
            __type);

  @override
  dynamic buildValue(DefaultGroupTensorDescriptor descriptor) {
    var delta = descriptor.getInput(_expectedInputName) -
        descriptor.getInput(_logitInputName);
    return (delta * delta) / 2;
  }
}

class BinaryCrossEntropyLossImpl extends DefaultGroupTensorBase
    implements BinaryCrossEntropyLoss {
  static const String __type = "BinaryCrossEntropyLoss";

  static const String _expectedInputName = "expected";
  static const String _logitInputName = "logit";

  BinaryCrossEntropyLossImpl(expected, logit, {String name})
      : super({_expectedInputName: expected, _logitInputName: logit}, name,
            __type);

  @override
  dynamic buildValue(DefaultGroupTensorDescriptor descriptor) =>
      -(descriptor.getInput(_expectedInputName) *
              new Log(descriptor.getInput(_logitInputName)) +
          (-descriptor.getInput(_expectedInputName) + 1) *
              new Log(-descriptor.getInput(_logitInputName) + 1));
}

class BinaryCrossEntropyWithLogitLossImpl extends DefaultGroupTensorBase
    implements BinaryCrossEntropyWithLogitLoss {
  static const String __type = "BinaryCrossEntropyWithLogitLoss";

  static const String _expectedInputName = "expected";
  static const String _logitInputName = "logit";

  BinaryCrossEntropyWithLogitLossImpl(expected, logit, {String name})
      : super({_expectedInputName: expected, _logitInputName: logit}, name,
            __type);

  @override
  dynamic buildValue(DefaultGroupTensorDescriptor descriptor) =>
      new Relu(descriptor.getInput(_logitInputName)) -
      descriptor.getInput(_logitInputName) *
          descriptor.getInput(_expectedInputName) +
      new Log(new Exp(-new Abs(descriptor.getInput(_logitInputName))) + 1);
}
