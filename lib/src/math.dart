// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "tensor.dart";

import "impl/math.dart";

// CORE MATH

abstract class Adds implements Tensor {
  factory Adds(List inputs, {String name}) => new AddsImpl(inputs, name: name);
}

abstract class Add implements Tensor {
  factory Add(input1, input2, {String name}) =>
      new AddImpl(input1, input2, name: name);
}

abstract class Sub implements Tensor {
  factory Sub(input1, input2, {String name}) =>
      new SubImpl(input1, input2, name: name);
}

abstract class Neg implements Tensor {
  factory Neg(input, {String name}) => new NegImpl(input, name: name);
}

abstract class Mul implements Tensor {
  factory Mul(input1, input2, {String name}) =>
      new MulImpl(input1, input2, name: name);
}

abstract class Div implements Tensor {
  factory Div(numerator, denominator, {String name}) =>
      new DivImpl(numerator, denominator, name: name);
}

abstract class Reciprocal implements Tensor {
  factory Reciprocal(input, {String name}) =>
      new ReciprocalImpl(input, name: name);
}

abstract class Sqrt implements Tensor {
  factory Sqrt(input, {String name}) => new SqrtImpl(input, name: name);
}

abstract class Pow implements Tensor {
  factory Pow(input, num exponent, {String name}) => new PowImpl(input, exponent, name: name);
}

abstract class Exp implements Tensor {
  factory Exp(input, {String name}) => new ExpImpl(input, name: name);
}

abstract class Log implements Tensor {
  factory Log(input, {String name}) => new LogImpl(input, name: name);
}

abstract class Abs implements Tensor {
  factory Abs(input, {String name}) => new AbsImpl(input, name: name);
}

// MATRIX MATH

abstract class MatMul implements Tensor {
  factory MatMul(input1, input2, {String name}) =>
      new MatMulImpl(input1, input2, name: name);
}

abstract class Transpose implements Tensor {
  factory Transpose(input, {List<int> permutationAxis, String name}) =>
      new TransposeImpl(input, permutationAxis: permutationAxis, name: name);
}

// ACTIVATION

abstract class Sigmoid implements Tensor {
  factory Sigmoid(input, {String name}) => new SigmoidImpl(input, name: name);
}

abstract class Relu implements Tensor {
  factory Relu(input, {String name}) => new ReluImpl(input, name: name);
}

abstract class Softmax implements Tensor {
  factory Softmax(input, {String name}) => new SoftmaxImpl(input, name: name);
}

abstract class Tanh implements Tensor {
  factory Tanh(input, {String name}) => new TanhImpl(input, name: name);
}

// LOGIC

abstract class IsEqual implements Tensor {
  factory IsEqual(input1, input2, {String name}) =>
      new IsEqualImpl(input1, input2, name: name);
}

abstract class IsNotEqual implements Tensor {
  factory IsNotEqual(input1, input2, {String name}) =>
      new IsNotEqualImpl(input1, input2, name: name);
}

abstract class IsLess implements Tensor {
  factory IsLess(input1, input2, {String name}) =>
      new LessImpl(input1, input2, name: name);
}

abstract class IsLessOrEqual implements Tensor {
  factory IsLessOrEqual(input1, input2, {String name}) =>
      new IsLessOrEqualImpl(input1, input2, name: name);
}

abstract class IsGreater implements Tensor {
  factory IsGreater(input1, input2, {String name}) =>
      new GreaterImpl(input1, input2, name: name);
}

abstract class IsGreaterOrEqual implements Tensor {
  factory IsGreaterOrEqual(input1, input2, {String name}) =>
      new IsGreaterOrEqualImpl(input1, input2, name: name);
}

abstract class Select implements Tensor {
  factory Select(conditionInput, thenInput, elseInput, {String name}) =>
      new SelectImpl(conditionInput, thenInput, elseInput, name: name);
}

// REDUCE

abstract class ReduceMean implements Tensor {
  factory ReduceMean(input,
          {List<int> reductionAxis,
          bool keepDimensions = false,
          String name}) =>
      new ReduceMeanImpl(input, reductionAxis: reductionAxis, name: name);
}

abstract class ReduceSum implements Tensor {
  factory ReduceSum(input,
          {List<int> reductionAxis,
          bool keepDimensions = false,
          String name}) =>
      new ReduceSumImpl(input,
          reductionAxis: reductionAxis,
          keepDimensions: keepDimensions,
          name: name);
}

abstract class ReduceMax implements Tensor {
  factory ReduceMax(input,
          {List<int> reductionAxis,
          bool keepDimensions = false,
          String name}) =>
      new ReduceMaxImpl(input,
          reductionAxis: reductionAxis,
          keepDimensions: keepDimensions,
          name: name);
}

// ARG

abstract class ArgMax implements Tensor {
  factory ArgMax(input, {int axis, String name}) =>
      new ArgMaxImpl(input, axis: axis, name: name);
}

// CONVOLUTION

abstract class Convolution2d implements Tensor {
  factory Convolution2d(input,
          {kernel, int heightStride = 1, int widthStride = 1, String name}) =>
      new Convolution2dImpl(input,
          kernel: kernel,
          heightStride: heightStride,
          widthStride: widthStride,
          name: name);
}

abstract class MaxPool implements Tensor {
  factory MaxPool(input, {int blockHeight, int blockWidth, String name}) =>
      new MaxPoolImpl(input,
          blockHeight: blockHeight, blockWidth: blockWidth, name: name);
}

abstract class Reshape implements Tensor {
  factory Reshape(input, {List<int> newDimensions, String name}) =>
      new ReshapeImpl(input, newDimensions: newDimensions, name: name);
}

// LOSS

abstract class Loss2 implements Tensor {
  factory Loss2(labels, logits, {String name}) =>
      new Loss2Impl(labels, logits, name: name);
}

abstract class SigmoidCrossEntropyWithLogits implements Tensor {
  factory SigmoidCrossEntropyWithLogits(labels, logits, {String name}) =>
      new SigmoidCrossEntropyWithLogitsImpl(labels, logits, name: name);
}

abstract class SoftmaxCrossEntropyWithLogits implements Tensor {
  factory SoftmaxCrossEntropyWithLogits(labels, logits, {String name}) =>
      new SoftmaxCrossEntropyWithLogitsImpl(labels, logits, name: name);
}
