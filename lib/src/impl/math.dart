// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math" as math;
import "dart:typed_data";

import "package:collection/collection.dart";

import "package:tensor_math/tensor_math.dart" as tm;

import "../operation.dart";
import "../tensor.dart";
import "../group.dart";
import "../math.dart";

final _zeroFloat = new Float32x4.zero();
final _oneFloat = new Float32x4.splat(1.0);

final _zeroInt = new Int32x4(0, 0, 0, 0);
final _oneInt = new Int32x4(1, 1, 1, 1);

List<int> calculateReductionBroadcastGradientAxis(
    tm.NDShape shape1, tm.NDShape shape2) {
  var broadcastedShape = shape1.broadcast(shape2);

  var dimensions = [];

  var shapeDelta = broadcastedShape.dimensionCount - shape1.dimensionCount ??
      broadcastedShape.dimensionCount;

  for (var i = 0; i < broadcastedShape.dimensionCount; i++) {
    var value = i >= shapeDelta ? shape1[i - shapeDelta] : null;
    if (value == null || (value == 1 && broadcastedShape[i] > 1)) {
      dimensions.add(i);
    }
  }

  return dimensions;
}

List<int> calculateMatMulGradientPermutationAxis(tm.NDShape shape) =>
    new List.generate(
        shape.dimensionCount,
        (index) => index < shape.dimensionCount - 2
            ? index
            : (index == shape.dimensionCount - 2 ? index + 1 : index - 1));

// TODO calcolo del gradiente
class AddsImpl extends DefaultTensorBase implements Adds {
  static const String __type = "Adds";

  static const String _inputsInputName = "inputs";

  AddsImpl(List inputs, {String name})
      : super(
            inputs: mapMap(inputs.asMap(),
                key: (index, _) => "$_inputsInputName${index + 1}"),
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) {
    return descriptor.isEvaluatingDescriptor
        ? tm.addDescriptors(
            descriptor.inputNames.map<tm.NDDescriptor>((inputName) {
            tm.NDDescriptor shape = descriptor.getInputValue(inputName);

            return shape;
          }))
        : tm.adds(descriptor.inputNames
            .map((inputName) => descriptor.getInputValue(inputName)));
  }
}

class AddImpl extends DefaultDifferentiableTensorBase implements Add {
  static const String __type = "Add";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  AddImpl(input1, input2, {String name})
      : super(
            inputs: {_input1InputName: input1, _input2InputName: input2},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
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
            inputs: {_input1InputName: input1, _input2InputName: input2},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
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
            inputs: {_input1InputName: input1, _input2InputName: input2},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
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

class NegImpl extends DefaultDifferentiableTensorBase implements Neg {
  static const String __type = "Neg";

  static const String _inputInputName = "input";

  NegImpl(input, {String name})
      : super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName).neg();

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) =>
            -descriptor.backPropagatedGradientValue);
  }
}

class DivImpl extends DefaultDifferentiableTensorBase implements Div {
  static const String __type = "Div";

  static const String _numeratorInputName = "numerator";
  static const String _denominatorInputName = "denominator";

  DivImpl(numerator, denominator, {String name})
      : super(inputs: {
          _numeratorInputName: numerator,
          _denominatorInputName: denominator
        }, operationName: name, type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_numeratorInputName)
      .div(descriptor.getInputValue(_denominatorInputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_numeratorInputName,
        (TensorGradientDescriptor descriptor) {
      print(descriptor.backPropagatedGradientValue);
      print(descriptor.getInputValue(_denominatorInputName));

      return (descriptor.backPropagatedGradientValue /
              descriptor.getInputValue(_denominatorInputName))
          .reduceSum(
              reductionAxis: calculateReductionBroadcastGradientAxis(
                  descriptor.getInputValue(_numeratorInputName).shape,
                  descriptor.getInputValue(_denominatorInputName).shape))
          .reshape(
              newDimensions: descriptor
                  .getInputValue(_numeratorInputName)
                  .shape
                  .dimensions);
    });

    descriptor.setOutputGradient(_denominatorInputName,
        (TensorGradientDescriptor descriptor) {
      if (dataType.isBlocked) {
        return descriptor.backPropagatedGradientValue
            .elementWiseTernaryOperation(descriptor.outputValue,
                descriptor.getInputValue(_denominatorInputName),
                resultDataType: descriptor.backPropagatedGradientValue.dataType,
                ternaryOperation:
                    (Float32x4 bv, Float32x4 dv, Float32x4 d, valueCount) {
          var resultValue = bv / d;

          switch (valueCount) {
            case 3:
              resultValue = new Float32x4(
                  resultValue.x, resultValue.y, resultValue.z, 0.0);
              break;
            case 2:
              resultValue =
                  new Float32x4(resultValue.x, resultValue.y, 0.0, 0.0);
              break;
            case 1:
              resultValue = new Float32x4(resultValue.x, 0.0, 0.0, 0.0);
              break;
          }

          return -dv * resultValue;
        });
      } else {
        return descriptor.backPropagatedGradientValue
            .elementWiseTernaryOperation(descriptor.outputValue,
                descriptor.getInputValue(_denominatorInputName),
                resultDataType: descriptor.backPropagatedGradientValue.dataType,
                ternaryOperation:
                    (double bv, double dv, double d, valueCount) =>
                        -dv * (bv / d));
      }
    });
  }
}

class ReciprocalImpl extends DefaultDifferentiableTensorBase
    implements Reciprocal {
  static const String __type = "Reciprocal";

  static const String _inputInputName = "input";

  ReciprocalImpl(input, {String name})
      : super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName).reciprocal();

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_inputInputName,
        (TensorGradientDescriptor descriptor) {
      if (dataType.isBlocked) {
        return descriptor.backPropagatedGradientValue
            .elementWiseTernaryOperation(descriptor.outputValue,
                descriptor.getInputValue(_inputInputName),
                resultDataType: descriptor.backPropagatedGradientValue.dataType,
                ternaryOperation:
                    (Float32x4 bv, Float32x4 dv, Float32x4 d, valueCount) {
          var resultValue = bv / d;

          switch (valueCount) {
            case 3:
              resultValue = new Float32x4(
                  resultValue.x, resultValue.y, resultValue.z, 0.0);
              break;
            case 2:
              resultValue =
                  new Float32x4(resultValue.x, resultValue.y, 0.0, 0.0);
              break;
            case 1:
              resultValue = new Float32x4(resultValue.x, 0.0, 0.0, 0.0);
              break;
          }

          return -dv * resultValue;
        });
      } else {
        return descriptor.backPropagatedGradientValue
            .elementWiseTernaryOperation(descriptor.outputValue,
                descriptor.getInputValue(_inputInputName),
                resultDataType: descriptor.backPropagatedGradientValue.dataType,
                ternaryOperation:
                    (double bv, double dv, double d, valueCount) =>
                        -dv * (bv / d));
      }
    });
  }
}

class ExpImpl extends DefaultDifferentiableTensorBase implements Exp {
  static const String __type = "Exp";

  static const String _inputInputName = "input";

  ExpImpl(input, {String name})
      : super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName).exp();

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) =>
            descriptor.backPropagatedGradientValue * descriptor.outputValue);
  }
}

class LogImpl extends DefaultDifferentiableTensorBase implements Log {
  static const String __type = "Log";

  static const String _inputInputName = "input";

  LogImpl(input, {String name})
      : super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
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

// TODO testare
class AbsImpl extends DefaultDifferentiableTensorBase implements Abs {
  static const String __type = "Abs";

  static const String _inputInputName = "input";

  AbsImpl(input, {String name})
      : super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName).abs();

  @override
  // TODO ottimizzare con operazione unica
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_inputInputName,
        (TensorGradientDescriptor descriptor) {
      print("ottimizzare abs");

      return descriptor
          .getInputValue(_inputInputName)
          .isGreaterOrEqual(0.0)
          .select(descriptor.backPropagatedGradientValue,
              -descriptor.backPropagatedGradientValue);
    });
  }
}

// MATRIX MATH

class MatMulImpl extends DefaultDifferentiableTensorBase implements MatMul {
  static const String __type = "MatMul";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  MatMulImpl(input1, input2, {String name})
      : super(
            inputs: {_input1InputName: input1, _input2InputName: input2},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .matMul(descriptor.getInputValue(_input2InputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_input1InputName,
        (TensorGradientDescriptor descriptor) {
      return descriptor.backPropagatedGradientValue.matMul(descriptor
          .getInputValue(_input2InputName)
          .transpose(
              permutationAxis: calculateMatMulGradientPermutationAxis(
                  descriptor.getInputValue(_input2InputName).shape)));
    });

    descriptor.setOutputGradient(_input2InputName,
        (TensorGradientDescriptor descriptor) {
      return descriptor
          .getInputValue(_input1InputName)
          .transpose(
              permutationAxis: calculateMatMulGradientPermutationAxis(
                  descriptor.getInputValue(_input1InputName).shape))
          .matMul(descriptor.backPropagatedGradientValue);
    });
  }
}

class TransposeImpl extends DefaultDifferentiableTensorBase
    implements Transpose {
  static const String __type = "Transpose";

  static const String _inputInputName = "input";

  List<int> _permutationAxis;

  TransposeImpl(input, {List<int> permutationAxis, String name})
      : this._permutationAxis = permutationAxis != null
            ? new List.unmodifiable(permutationAxis)
            : null,
        super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_inputInputName)
      .transpose(permutationAxis: _permutationAxis);

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    // TODO to implement TransposeImpl.buildDefaultGradients
    throw new UnimplementedError(
        "to implement TransposeImpl.buildDefaultGradients: $this");
  }
}

// ACTIVATION

// TODO testare
class SigmoidImpl extends DefaultDifferentiableTensorBase implements Sigmoid {
  static const String __type = "Sigmoid";

  static const String _inputInputName = "input";

  SigmoidImpl(input, {String name})
      : super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  // TODO ottimizzare con operazione unica
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) {
    print("ottimizzare sigmoid");

    // TODO attenzione al tipo della constante
    return (descriptor.getInputValue(_inputInputName).neg().exp() +
            descriptor.toNDObject(1.0))
        .reciprocal();
  }

  @override
  // TODO ottimizzare con operazione unica
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_inputInputName,
        (TensorGradientDescriptor descriptor) {
      print("ottimizzare sigmoid");

      // TODO attenzione al tipo della constante
      return descriptor.backPropagatedGradientValue *
          descriptor.outputValue *
          (new tm.NDArray(1.0) - descriptor.outputValue);
    });
  }
}

// TODO testare
class TanhImpl extends DefaultDifferentiableTensorBase implements Tanh {
  static const String __type = "Tanh";

  static const String _inputInputName = "input";

  TanhImpl(input, {String name})
      : super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  // TODO ottimizzare con operazione unica
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) {
    print("ottimizzare tanh");

    var e2x =
        (descriptor.getInputValue(_inputInputName) * descriptor.toNDObject(2.0))
            .exp();

    // TODO attenzione al tipo della constante
    return (e2x - descriptor.toNDObject(1.0)) /
        (e2x + descriptor.toNDObject(1.0));
  }

  @override
  // TODO ottimizzare con operazione unica
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_inputInputName,
        (TensorGradientDescriptor descriptor) {
      print("ottimizzare tanh");

      // TODO attenzione al tipo della constante
      return descriptor.backPropagatedGradientValue *
          (new tm.NDArray(1.0) -
              (descriptor.outputValue * descriptor.outputValue));
    });
  }
}

// TODO testare
class ReluImpl extends DefaultDifferentiableTensorBase implements Relu {
  static const String __type = "Relu";

  static const String _inputInputName = "input";

  ReluImpl(input, {String name})
      : super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) {
    if (descriptor.getInputValue(_inputInputName).dataType.isBlocked) {
      // TODO gestire caso float e int

      // TODO attenzione al tipo della constante

      return descriptor
          .getInputValue(_inputInputName)
          .elementWiseUnaryOperation(
              resultDataType:
                  descriptor.getInputValue(_inputInputName).dataType,
              unaryOperation: (Float32x4 value, valueCount) =>
                  value.max(_zeroFloat));
    } else if (descriptor.getInputValue(_inputInputName).dataType.isFloat) {
      return descriptor
          .getInputValue(_inputInputName)
          .elementWiseUnaryOperation(
              resultDataType:
                  descriptor.getInputValue(_inputInputName).dataType,
              unaryOperation: (value, valueCount) => value > 0.0 ? value : 0.0);
    } else if (descriptor.getInputValue(_inputInputName).dataType.isInteger) {
      return descriptor
          .getInputValue(_inputInputName)
          .elementWiseUnaryOperation(
              resultDataType:
                  descriptor.getInputValue(_inputInputName).dataType,
              unaryOperation: (value, valueCount) => value > 0 ? value : 0);
    } else {
      throw new UnsupportedError(
          "Relu on ${descriptor.getInputValue(_inputInputName).dataType} array type");
    }
  }

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_inputInputName,
        (TensorGradientDescriptor descriptor) {
      if (descriptor.getInputValue(_inputInputName).dataType.isBlocked) {
        // TODO gestire caso float e int

        // TODO attenzione al tipo della constante

        return descriptor.backPropagatedGradientValue
            .elementWiseBinaryOperation(
                descriptor.getInputValue(_inputInputName),
                resultDataType:
                    descriptor.getInputValue(_inputInputName).dataType,
                binaryOperation:
                    (Float32x4 value1, Float32x4 value2, valueCount) {
          // TODO attenzione clamp non va bene perchè risultato o zero o uno
          Float32x4 result = value2.clamp(_zeroFloat, _oneFloat);

          switch (valueCount) {
            case 4:
              result = new Float32x4(result.x, result.y, result.z, result.w);
              break;
            case 3:
              result = new Float32x4(result.x, result.y, result.z, 0.0);
              break;
            case 2:
              result = new Float32x4(result.x, result.y, 0.0, 0.0);
              break;
            case 1:
              result = new Float32x4(result.x, 0.0, 0.0, 0.0);
              break;
          }

          return value1 * result;
        });
      } else if (descriptor.getInputValue(_inputInputName).dataType.isFloat) {
        return descriptor.backPropagatedGradientValue
            .elementWiseBinaryOperation(
                descriptor.getInputValue(_inputInputName),
                resultDataType:
                    descriptor.getInputValue(_inputInputName).dataType,
                binaryOperation: (value1, value2, valueCount) =>
                    value2 > 0.0 ? value1 : 0.0);
      } else if (descriptor.getInputValue(_inputInputName).dataType.isInteger) {
        return descriptor.backPropagatedGradientValue
            .elementWiseBinaryOperation(
                descriptor.getInputValue(_inputInputName),
                resultDataType:
                    descriptor.getInputValue(_inputInputName).dataType,
                binaryOperation: (value1, value2, valueCount) =>
                    value2 > 0 ? value1 : 0);
      } else {
        throw new UnsupportedError(
            "Relu on ${descriptor.getInputValue(_inputInputName).dataType} array type");
      }
    });
  }
}

// TODO testare
class SoftmaxImpl extends DefaultTensorBase implements Softmax {
  static const String __type = "Softmax";

  static const String _inputInputName = "input";

  SoftmaxImpl(input, {String name})
      : super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
      _softmax(descriptor.getInputValue(_inputInputName));
}

// LOGIC

class IsEqualImpl extends DefaultTensorBase implements IsEqual {
  static const String __type = "IsEqual";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  IsEqualImpl(input1, input2, {String name})
      : super(
            inputs: {_input1InputName: input1, _input2InputName: input2},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .isEqual(descriptor.getInputValue(_input2InputName));
}

class IsNotEqualImpl extends DefaultTensorBase implements IsNotEqual {
  static const String __type = "IsNotEqual";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  IsNotEqualImpl(input1, input2, {String name})
      : super(
            inputs: {_input1InputName: input1, _input2InputName: input2},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .isNotEqual(descriptor.getInputValue(_input2InputName));
}

class LessImpl extends DefaultTensorBase implements IsLess {
  static const String __type = "Less";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  LessImpl(input1, input2, {String name})
      : super(
            inputs: {_input1InputName: input1, _input2InputName: input2},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .isLess(descriptor.getInputValue(_input2InputName));
}

class IsLessOrEqualImpl extends DefaultTensorBase implements IsLessOrEqual {
  static const String __type = "IsLessOrEqual";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  IsLessOrEqualImpl(input1, input2, {String name})
      : super(
            inputs: {_input1InputName: input1, _input2InputName: input2},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .isLessOrEqual(descriptor.getInputValue(_input2InputName));
}

class GreaterImpl extends DefaultTensorBase implements IsGreater {
  static const String __type = "Greater";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  GreaterImpl(input1, input2, {String name})
      : super(
            inputs: {_input1InputName: input1, _input2InputName: input2},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .isGreater(descriptor.getInputValue(_input2InputName));
}

class IsGreaterOrEqualImpl extends DefaultTensorBase
    implements IsGreaterOrEqual {
  static const String __type = "IsGreaterOrEqual";

  static const String _input1InputName = "input1";
  static const String _input2InputName = "input2";

  IsGreaterOrEqualImpl(input1, input2, {String name})
      : super(
            inputs: {_input1InputName: input1, _input2InputName: input2},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) => descriptor
      .getInputValue(_input1InputName)
      .isGreaterOrEqual(descriptor.getInputValue(_input2InputName));
}

// TODO testare
class SelectImpl extends DefaultDifferentiableTensorBase implements Select {
  static const String __type = "Select";

  static const String _conditionInputInputName = "conditionInput";
  static const String _thenInputInputName = "thenInput";
  static const String _elseInputInputName = "elseInput";

  SelectImpl(conditionInput, thenInput, elseInput, {String name})
      : super(inputs: {
          _conditionInputInputName: conditionInput,
          _thenInputInputName: thenInput,
          _elseInputInputName: elseInput
        }, operationName: name, type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_conditionInputInputName).select(
          descriptor.getInputValue(_thenInputInputName),
          descriptor.getInputValue(_elseInputInputName));

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    // TODO attenzione al tipo della constante

    descriptor.setOutputGradient(
        _thenInputInputName,
        (TensorGradientDescriptor descriptor) => descriptor
            .getInputValue(_conditionInputInputName)
            .select(descriptor.backPropagatedGradientValue, 0.0)
            .reshape(
                newDimensions: descriptor
                    .getInputValue(_thenInputInputName)
                    .shape
                    .dimensions));

    // TODO attenzione al tipo della constante

    descriptor.setOutputGradient(
        _elseInputInputName,
        (TensorGradientDescriptor descriptor) => descriptor
            .getInputValue(_conditionInputInputName)
            .select(0.0, descriptor.backPropagatedGradientValue)
            .reshape(
                newDimensions: descriptor
                    .getInputValue(_thenInputInputName)
                    .shape
                    .dimensions));
  }
}

// REDUCE

class ReduceSumImpl extends DefaultDifferentiableTensorBase
    implements ReduceSum {
  static const String __type = "ReduceSum";

  static const String _inputInputName = "input";

  final List<int> _reductionAxis;

  final bool _keepDimensions;

  ReduceSumImpl(input,
      {List<int> reductionAxis, bool keepDimensions = false, String name})
      : this._reductionAxis =
            reductionAxis != null ? new List.unmodifiable(reductionAxis) : null,
        this._keepDimensions = keepDimensions,
        super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName).reduceSum(
          reductionAxis: _reductionAxis, keepDimensions: _keepDimensions);

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_inputInputName,
        (TensorGradientDescriptor descriptor) {
      if (descriptor.getInputValue(_inputInputName).shape.dimensionCount >
          descriptor.backPropagatedGradientValue.shape.dimensionCount) {
        var newReductionAxis = new List.generate(
            descriptor.getInputValue(_inputInputName).shape.dimensionCount,
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

// TODO testare
class ReduceMeanImpl extends DefaultDifferentiableTensorBase
    implements ReduceMean {
  static const String __type = "ReduceMean";

  static const String _inputInputName = "input";

  final List<int> _reductionAxis;

  final bool _keepDimensions;

  ReduceMeanImpl(input,
      {List<int> reductionAxis, bool keepDimensions = false, String name})
      : this._reductionAxis =
            reductionAxis != null ? new List.unmodifiable(reductionAxis) : null,
        this._keepDimensions = keepDimensions,
        super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName).reduceMean(
          reductionAxis: _reductionAxis, keepDimensions: _keepDimensions);

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_inputInputName,
        (TensorGradientDescriptor descriptor) {
      if (descriptor.getInputValue(_inputInputName).shape.dimensionCount >
          descriptor.backPropagatedGradientValue.shape.dimensionCount) {
        var newReductionAxis = new List.generate(
            descriptor.getInputValue(_inputInputName).shape.dimensionCount,
            (index) => index);

        var dimensions = new List.from(
            descriptor.getInputValue(_inputInputName).shape.dimensions);
        var multiplies = new List.filled(dimensions.length, 1);
        for (var index in newReductionAxis) {
          multiplies[index] = dimensions[index];
          dimensions[index] = 1;
        }

        var factor = multiplies.reduce((total, value) => total * value);

        // TODO attenzione al tipo di factor (solo float ma blocked e non)
        return (descriptor.backPropagatedGradientValue / factor)
            .reshape(newDimensions: dimensions)
            .tile(multiplies);
      } else {
        return descriptor.backPropagatedGradientValue;
      }
    });
  }
}

class ReduceMaxImpl extends DefaultDifferentiableTensorBase
    implements ReduceMax {
  static const String __type = "ReduceMax";

  static const String _inputInputName = "input";

  final List<int> _reductionAxis;

  final bool _keepDimensions;

  ReduceMaxImpl(input,
      {List<int> reductionAxis, bool keepDimensions = false, String name})
      : this._reductionAxis =
            reductionAxis != null ? new List.unmodifiable(reductionAxis) : null,
        this._keepDimensions = keepDimensions,
        super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName).reduceMean(
          reductionAxis: _reductionAxis, keepDimensions: _keepDimensions);

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    // TODO to implement ReduceMaxImpl.buildDefaultGradients
    throw new UnimplementedError(
        "to implement ReduceMaxImpl.buildDefaultGradients: $this");
  }
}

// ARG

class ArgMaxImpl extends DefaultTensorBase implements ArgMax {
  static const String __type = "ArgMax";

  static const String _inputInputName = "input";

  final int _axis;

  ArgMaxImpl(input, {int axis, String name})
      : this._axis = axis,
        super(
            inputs: {_inputInputName: input},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName).argMax(axis: _axis);
}

// LOSS

// TODO testare
class Loss2Impl extends DefaultGroupTensorBase implements Loss2 {
  static const String __type = "Loss2";

  static const String _labelsInputName = "labels";
  static const String _logitsInputName = "logits";

  Loss2Impl(labels, logits, {String name})
      : super(
            inputs: {_labelsInputName: labels, _logitsInputName: logits},
            operationName: name,
            type: __type);

  @override
  Tensor buildValue(DefaultGroupTensorDescriptor descriptor) {
    // TODO attenzione al tipo della constante

    var delta = descriptor.getInput(_labelsInputName) -
        descriptor.getInput(_logitsInputName);
    return (delta * delta) / 2;
  }
}

// TODO testare
class SigmoidCrossEntropyWithLogitsImpl extends DefaultGroupTensorBase
    implements SigmoidCrossEntropyWithLogits {
  static const String __type = "SigmoidCrossEntropyWithLogits";

  static const String _labelsInputName = "labels";
  static const String _logitsInputName = "logits";

  SigmoidCrossEntropyWithLogitsImpl(labels, logits, {String name})
      : super(
            inputs: {_labelsInputName: labels, _logitsInputName: logits},
            operationName: name,
            type: __type);

  @override
  Tensor buildValue(DefaultGroupTensorDescriptor descriptor) =>
      // TODO attenzione al tipo della constante
      new Relu(descriptor.getInput(_logitsInputName)) -
      descriptor.getInput(_logitsInputName) *
          descriptor.getInput(_labelsInputName) +
      new Log(new Exp(-new Abs(descriptor.getInput(_logitsInputName))) + 1.0);
}

// TODO testare
class SoftmaxCrossEntropyWithLogitsImpl extends DefaultDifferentiableTensorBase
    implements SoftmaxCrossEntropyWithLogits {
  static const String __type = "SoftmaxCrossEntropyWithLogits";

  static const String _labelsInputName = "labels";
  static const String _logitsInputName = "logits";

  SoftmaxCrossEntropyWithLogitsImpl(labels, logits, {String name})
      : super(
            inputs: {_labelsInputName: labels, _logitsInputName: logits},
            operationName: name,
            type: __type);

  @override
  tm.NDObject computeValue(DefaultTensorDescriptor descriptor) {
    var sm = _softmax(descriptor.getInputValue(_logitsInputName));

    // TODO utilizzare funzioni utility del descrittore
    if (!descriptor.isEvaluatingDescriptor) {
      state["_softmax"] = sm;
    }

    // TODO check dei tipi

    var value;
    if (descriptor.getInputValue(_logitsInputName).dataType.isBlocked) {
      // TODO gestire caso float e int

      // TODO attenzione al tipo della constante

      value = sm.elementWiseBinaryOperation(
          descriptor.getInputValue(_labelsInputName),
          resultDataType: descriptor.getInputValue(_logitsInputName).dataType,
          binaryOperation: (Float32x4 value1, Float32x4 value2, valueCount) {
        Float32x4 result;
        switch (valueCount) {
          case 4:
            result = new Float32x4(math.log(value1.x), math.log(value1.y),
                math.log(value1.z), math.log(value1.w));
            break;
          case 3:
            result = new Float32x4(math.log(value1.x), math.log(value1.y),
                math.log(value1.z), 0.0);
            break;
          case 2:
            result =
                new Float32x4(math.log(value1.x), math.log(value1.y), 0.0, 0.0);
            break;
          case 1:
            result = new Float32x4(math.log(value1.x), 0.0, 0.0, 0.0);
            break;
        }

        return -result * value2;
      });
    } else {
      value = sm.elementWiseBinaryOperation(
          descriptor.getInputValue(_labelsInputName),
          resultDataType: descriptor.getInputValue(_logitsInputName).dataType,
          binaryOperation: (value1, value2, valueCount) =>
              -(math.log(value1) * value2));
    }

    return value.reduceSum(reductionAxis: [
      descriptor.getInputValue(_logitsInputName).shape.dimensionCount - 1
    ]);
  }

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_logitsInputName, (descriptor) {
      // TODO utilizzare funzioni utility del descrittore
      var sm = descriptor.toNDObject(state["_softmax"]);

      return sm - descriptor.getInputValue(_labelsInputName);
    });
  }
}

tm.NDObject _softmax(tm.NDObject value) {
  var r1 = value.reduceMax(
      reductionAxis: [value.shape.dimensionCount - 1], keepDimensions: true);

  var r2;
  if (value.dataType.isBlocked) {
    // TODO gestire caso float e int

    // TODO attenzione al tipo della constante

    r2 = value.elementWiseBinaryOperation(r1, resultDataType: value.dataType,
        binaryOperation: (Float32x4 value1, Float32x4 value2, valueCount) {
      var result = value1 - value2;

      switch (valueCount) {
        case 4:
          return new Float32x4(math.exp(result.x), math.exp(result.y),
              math.exp(result.z), math.exp(result.w));
        case 3:
          return new Float32x4(
              math.exp(result.x), math.exp(result.y), math.exp(result.z), 0.0);
        case 2:
          return new Float32x4(
              math.exp(result.x), math.exp(result.y), 0.0, 0.0);
        case 1:
          return new Float32x4(math.exp(result.x), 0.0, 0.0, 0.0);
      }
    });
  } else {
    r2 = value.elementWiseBinaryOperation(r1,
        resultDataType: value.dataType,
        binaryOperation: (value1, value2, valueCount) =>
            math.exp(value1 - value2));
  }

  return r2 /
      r2.reduceSum(
          reductionAxis: [value.shape.dimensionCount - 1],
          keepDimensions: true);
}
