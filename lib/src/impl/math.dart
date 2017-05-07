// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:collection/collection.dart";

import "package:tensor_math/tensor_math.dart" as math;

import "../operation.dart";
import "../tensor.dart";
import "../group.dart";
import "../math.dart";

List<int> calculateReductionBroadcastGradientAxis(
    math.NDShape shape1, math.NDShape shape2) {
  var broadcastedShape = shape1.broadcast(shape2);

  var dimensions = [];

  var shapeDelta = broadcastedShape.dimension - shape1.dimension;

  for (var i = 0; i < broadcastedShape.dimension; i++) {
    var value = i >= shapeDelta ? shape1[i - shapeDelta] : null;
    if (value == null || (value == 1 && broadcastedShape[i] > 1)) {
      dimensions.add(i);
    }
  }

  return dimensions;
}

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
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_input1InputName) +
      descriptor.getInputValue(_input2InputName);

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _input1InputName,
        (TensorGradientDescriptor descriptor) => descriptor
            .backPropagatedGradientValue
            .reduceSum(
                reductionAxis: calculateReductionBroadcastGradientAxis(
                    descriptor.getInputValue(_input1InputName).shape,
                    descriptor.getInputValue(_input2InputName).shape))
            .reshape(
                newDimensions: descriptor
                    .getInputValue(_input1InputName)
                    .shape
                    .dimensions));

    descriptor.setOutputGradient(
        _input2InputName,
        (TensorGradientDescriptor descriptor) => descriptor
            .backPropagatedGradientValue
            .reduceSum(
                reductionAxis: calculateReductionBroadcastGradientAxis(
                    descriptor.getInputValue(_input2InputName).shape,
                    descriptor.getInputValue(_input1InputName).shape))
            .reshape(
                newDimensions: descriptor
                    .getInputValue(_input2InputName)
                    .shape
                    .dimensions));
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
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_input1InputName) -
      descriptor.getInputValue(_input2InputName);

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _input1InputName,
        (TensorGradientDescriptor descriptor) => descriptor
            .backPropagatedGradientValue
            .reduceSum(
                reductionAxis: calculateReductionBroadcastGradientAxis(
                    descriptor.getInputValue(_input1InputName).shape,
                    descriptor.getInputValue(_input2InputName).shape))
            .reshape(
                newDimensions: descriptor
                    .getInputValue(_input1InputName)
                    .shape
                    .dimensions));

    descriptor.setOutputGradient(
        _input2InputName,
        (TensorGradientDescriptor descriptor) => -descriptor
            .backPropagatedGradientValue
            .reduceSum(
                reductionAxis: calculateReductionBroadcastGradientAxis(
                    descriptor.getInputValue(_input2InputName).shape,
                    descriptor.getInputValue(_input1InputName).shape))
            .reshape(
                newDimensions: descriptor
                    .getInputValue(_input2InputName)
                    .shape
                    .dimensions));
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
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_input1InputName) *
      descriptor.getInputValue(_input2InputName);

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _input1InputName,
        (TensorGradientDescriptor descriptor) =>
            (descriptor.backPropagatedGradientValue *
                    descriptor.getInputValue(_input2InputName))
                .reduceSum(
                    reductionAxis: calculateReductionBroadcastGradientAxis(
                        descriptor.getInputValue(_input1InputName).shape,
                        descriptor.getInputValue(_input2InputName).shape))
                .reshape(
                    newDimensions: descriptor
                        .getInputValue(_input1InputName)
                        .shape
                        .dimensions));

    descriptor.setOutputGradient(
        _input2InputName,
        (TensorGradientDescriptor descriptor) =>
            (descriptor.getInputValue(_input1InputName) *
                    descriptor.backPropagatedGradientValue)
                .reduceSum(
                    reductionAxis: calculateReductionBroadcastGradientAxis(
                        descriptor.getInputValue(_input2InputName).shape,
                        descriptor.getInputValue(_input1InputName).shape))
                .reshape(
                    newDimensions: descriptor
                        .getInputValue(_input2InputName)
                        .shape
                        .dimensions));
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
  dynamic computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_numeratorInputName)
      .div(descriptor.getInputValue(_denominatorInputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _numeratorInputName,
        (TensorGradientDescriptor descriptor) =>
            (descriptor.backPropagatedGradientValue /
                    descriptor.getInputValue(_denominatorInputName))
                .reduceSum(
                    reductionAxis: calculateReductionBroadcastGradientAxis(
                        descriptor.getInputValue(_numeratorInputName).shape,
                        descriptor.getInputValue(_denominatorInputName).shape))
                .reshape(
                    newDimensions: descriptor
                        .getInputValue(_numeratorInputName)
                        .shape
                        .dimensions));

    descriptor.setOutputGradient(
        _denominatorInputName,
        (TensorGradientDescriptor descriptor) =>
            (descriptor.backPropagatedGradientValue *
                    ((-descriptor.getInputValue(_numeratorInputName) /
                            descriptor.getInputValue(_denominatorInputName)) /
                        descriptor.getInputValue(_denominatorInputName)))
                .reduceSum(
                    reductionAxis: calculateReductionBroadcastGradientAxis(
                        descriptor.getInputValue(_denominatorInputName).shape,
                        descriptor.getInputValue(_numeratorInputName).shape))
                .reshape(
                    newDimensions: descriptor
                        .getInputValue(_denominatorInputName)
                        .shape
                        .dimensions));
  }
}

class NegImpl extends DefaultDifferentiableTensorBase implements Neg {
  static const String __type = "Neg";

  static const String _inputInputName = "input";

  NegImpl(input, {String name}) : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName).neg();

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) =>
            -descriptor.backPropagatedGradientValue);
  }
}

class InvImpl extends DefaultDifferentiableTensorBase implements Inv {
  static const String __type = "Inv";

  static const String _inputInputName = "input";

  InvImpl(input, {String name}) : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName).inv();

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) =>
            (-descriptor.backPropagatedGradientValue /
                descriptor.getInputValue(_inputInputName) /
                descriptor.getInputValue(_inputInputName)));
  }
}

class ExpImpl extends DefaultDifferentiableTensorBase implements Exp {
  static const String __type = "Exp";

  static const String _inputInputName = "input";

  ExpImpl(input, {String name}) : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName).exp();

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) =>
            descriptor.backPropagatedGradientValue *
            descriptor.getInputValue(_inputInputName).exp());
  }
}

class LogImpl extends DefaultDifferentiableTensorBase implements Log {
  static const String __type = "Log";

  static const String _inputInputName = "input";

  LogImpl(input, {String name}) : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName).log();

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) =>
            descriptor.backPropagatedGradientValue /
            descriptor.getInputValue(_inputInputName));
  }
}

class AbsImpl extends DefaultDifferentiableTensorBase implements Abs {
  static const String __type = "Abs";

  static const String _inputInputName = "input";

  AbsImpl(input, {String name}) : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName).abs();

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) => descriptor
            .getInputValue(_inputInputName)
            .greaterOrEquals(0)
            .select(descriptor.backPropagatedGradientValue,
                -descriptor.backPropagatedGradientValue));
  }
}

class SigmoidImpl extends DefaultDifferentiableTensorBase implements Sigmoid {
  static const String __type = "Sigmoid";

  static const String _inputInputName = "input";

  SigmoidImpl(input, {String name})
      : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      (descriptor.getInputValue(_inputInputName).neg().exp() + 1).inv();

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) =>
            descriptor.backPropagatedGradientValue *
            descriptor.outputValue *
            (new math.NDArray(1) - descriptor.outputValue));
  }
}

class TanhImpl extends DefaultDifferentiableTensorBase implements Tanh {
  static const String __type = "Tanh";

  static const String _inputInputName = "input";

  TanhImpl(input, {String name})
      : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) {
    var e2x = (descriptor.getInputValue(_inputInputName) * 2).exp();

    return (e2x - 1) / (e2x + 1);
  }

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) =>
            descriptor.backPropagatedGradientValue *
            (new math.NDArray(1) -
                (descriptor.outputValue * descriptor.outputValue)));
  }
}

class ReduceSumImpl extends DefaultDifferentiableTensorBase
    implements ReduceSum {
  static const String __type = "ReduceSum";

  static const String _inputInputName = "input";

  final List<int> _reductionAxis;

  ReduceSumImpl(input, {List<int> reductionAxis, String name})
      : this._reductionAxis = reductionAxis,
        super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_inputInputName)
      .reduceSum(reductionAxis: _reductionAxis);

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_inputInputName,
        (TensorGradientDescriptor descriptor) {
      if (descriptor.getInputValue(_inputInputName).shape.dimension >
          descriptor.backPropagatedGradientValue.shape.dimension) {
        var newReductionAxis = new List.generate(
            descriptor.getInputValue(_inputInputName).shape.dimension,
            (index) => index);

        var dimensions =
            descriptor.getInputValue(_inputInputName).shape.dimensions;
        var multiplies = new List.filled(dimensions.length, 1);
        for (var index in newReductionAxis) {
          multiplies[index] = dimensions[index];
          dimensions[index] = 1;
        }

        return descriptor.backPropagatedGradientValue
            .reshape(newDimensions: dimensions)
            .tile(multiplies);
      } else {
        return descriptor.backPropagatedGradientValue;
      }
    });
  }
}

class ReduceMeanImpl extends DefaultDifferentiableTensorBase
    implements ReduceMean {
  static const String __type = "ReduceMean";

  static const String _inputInputName = "input";

  List<int> _reductionAxis;

  ReduceMeanImpl(input, {List<int> reductionAxis, String name})
      : this._reductionAxis = reductionAxis,
        super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_inputInputName)
      .reduceMean(reductionAxis: _reductionAxis);

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_inputInputName,
        (TensorGradientDescriptor descriptor) {
      if (descriptor.getInputValue(_inputInputName).shape.dimension >
          descriptor.backPropagatedGradientValue.shape.dimension) {
        var newReductionAxis = new List.generate(
            descriptor.getInputValue(_inputInputName).shape.dimension,
            (index) => index);

        var dimensions =
            descriptor.getInputValue(_inputInputName).shape.dimensions;
        var multiplies = new List.filled(dimensions.length, 1);
        for (var index in newReductionAxis) {
          multiplies[index] = dimensions[index];
          dimensions[index] = 1;
        }

        var factor = multiplies.reduce((total, value) => total * value);

        return (descriptor.backPropagatedGradientValue / factor)
            .reshape(newDimensions: dimensions)
            .tile(multiplies);
      } else {
        return descriptor.backPropagatedGradientValue;
      }
    });
  }
}

class MatMulImpl extends DefaultDifferentiableTensorBase implements MatMul {
  static const String __type = "MatMul";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  MatMulImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .matMul(descriptor.getInputValue(_input2InputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    // TODO rivedere matMul con transposeA e transposeB
    descriptor.setOutputGradient(
        _input1InputName,
        (TensorGradientDescriptor descriptor) => descriptor
            .backPropagatedGradientValue
            .matMul(descriptor.getInputValue(_input2InputName).transpose(
                    permutationAxis: new List.generate(
                        descriptor
                            .getInputValue(_input2InputName)
                            .shape
                            .dimension, (index) {
                  return index <
                          descriptor
                                  .getInputValue(_input2InputName)
                                  .shape
                                  .dimension -
                              2
                      ? index
                      : (index ==
                              descriptor
                                      .getInputValue(_input2InputName)
                                      .shape
                                      .dimension -
                                  2
                          ? index + 1
                          : index - 1);
                }))));

    // TODO rivedere matMul con transposeA e transposeB
    descriptor.setOutputGradient(
        _input2InputName,
        (TensorGradientDescriptor descriptor) => descriptor
            .getInputValue(_input1InputName)
            .transpose(
                permutationAxis: new List.generate(
                    descriptor.getInputValue(_input2InputName).shape.dimension,
                    (index) {
              return index <
                      descriptor
                              .getInputValue(_input2InputName)
                              .shape
                              .dimension -
                          2
                  ? index
                  : (index ==
                          descriptor
                                  .getInputValue(_input2InputName)
                                  .shape
                                  .dimension -
                              2
                      ? index + 1
                      : index - 1);
            }))
            .matMul(descriptor.backPropagatedGradientValue));
  }
}

class EqualsImpl extends DefaultTensorBase implements Equals {
  static const String __type = "Equals";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  EqualsImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .equals(descriptor.getInputValue(_input2InputName));
}

class NotEqualsImpl extends DefaultTensorBase implements NotEquals {
  static const String __type = "NotEquals";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  NotEqualsImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .notEquals(descriptor.getInputValue(_input2InputName));
}

class LessImpl extends DefaultTensorBase implements Less {
  static const String __type = "Less";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  LessImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .less(descriptor.getInputValue(_input2InputName));
}

class LessOrEqualsImpl extends DefaultTensorBase implements LessOrEquals {
  static const String __type = "LessOrEquals";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  LessOrEqualsImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .lessOrEquals(descriptor.getInputValue(_input2InputName));
}

class GreaterImpl extends DefaultTensorBase implements Greater {
  static const String __type = "Greater";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  GreaterImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .greater(descriptor.getInputValue(_input2InputName));
}

class GreaterOrEqualsImpl extends DefaultTensorBase implements GreaterOrEquals {
  static const String __type = "GreaterOrEquals";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  GreaterOrEqualsImpl(input1, input2, {String name})
      : super(
            {_input1InputName: input1, _input2InputName: input2}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .greaterOrEquals(descriptor.getInputValue(_input2InputName));
}

class ReluImpl extends DefaultDifferentiableTensorBase implements Relu {
  static const String __type = "Relu";

  static const String _inputInputName = "input";

  ReluImpl(input, {String name})
      : super({_inputInputName: input}, name, __type);

  @override
  dynamic computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_inputInputName)
      .greaterOrEquals(0)
      .select(descriptor.getInputValue(_inputInputName), 0);

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_inputInputName,
        (TensorGradientDescriptor descriptor) {
      return descriptor.backPropagatedGradientValue *
          descriptor
              .getInputValue(_inputInputName)
              .greaterOrEquals(0)
              .select(1, 0);
    });
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
  dynamic computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_conditionInputInputName).select(
          descriptor.getInputValue(_thenInputInputName),
          descriptor.getInputValue(_elseInputInputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _thenInputInputName,
        (TensorGradientDescriptor descriptor) =>
            descriptor.backPropagatedGradientValue *
            descriptor.getInputValue(_conditionInputInputName).select(1, 0));

    descriptor.setOutputGradient(
        _elseInputInputName,
        (TensorGradientDescriptor descriptor) =>
            descriptor.backPropagatedGradientValue *
            descriptor.getInputValue(_conditionInputInputName).select(0, 1));
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

class SigmoidCrossEntropyLossImpl extends DefaultGroupTensorBase
    implements SigmoidCrossEntropyLoss {
  static const String __type = "SigmoidCrossEntropyLoss";

  static const String _expectedInputName = "expected";
  static const String _logitInputName = "logit";

  SigmoidCrossEntropyLossImpl(expected, logit, {String name})
      : super({_expectedInputName: expected, _logitInputName: logit}, name,
            __type);

  @override
  dynamic buildValue(DefaultGroupTensorDescriptor descriptor) =>
      -(descriptor.getInput(_expectedInputName) *
              new Log(descriptor.getInput(_logitInputName)) +
          (-descriptor.getInput(_expectedInputName) + 1) *
              new Log(-descriptor.getInput(_logitInputName) + 1));
}

class SigmoidCrossEntropyWithLogitLossImpl extends DefaultGroupTensorBase
    implements SigmoidCrossEntropyWithLogitLoss {
  static const String __type = "SigmoidCrossEntropyWithLogitLoss";

  static const String _expectedInputName = "expected";
  static const String _logitInputName = "logit";

  SigmoidCrossEntropyWithLogitLossImpl(expected, logit, {String name})
      : super({_expectedInputName: expected, _logitInputName: logit}, name,
            __type);

  @override
  dynamic buildValue(DefaultGroupTensorDescriptor descriptor) =>
      new Relu(descriptor.getInput(_logitInputName)) -
      descriptor.getInput(_logitInputName) *
          descriptor.getInput(_expectedInputName) +
      new Log(new Exp(-new Abs(descriptor.getInput(_logitInputName))) + 1);
}
