// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "tensor.dart";

import "impl/math.dart";

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

abstract class MatMul implements Tensor {
  factory MatMul(input1, input2, {String name}) =>
      new MatMulImpl(input1, input2, name: name);
}

abstract class Div implements Tensor {
  factory Div(numerator, denominator, {String name}) =>
      new DivImpl(numerator, denominator, name: name);
}

abstract class Inv implements Tensor {
  factory Inv(input, {String name}) => new InvImpl(input, name: name);
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

abstract class Sigmoid implements Tensor {
  factory Sigmoid(input, {String name}) => new SigmoidImpl(input, name: name);
}

abstract class Tanh implements Tensor {
  factory Tanh(input, {String name}) => new TanhImpl(input, name: name);
}

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

abstract class Relu implements Tensor {
  factory Relu(input, {String name}) => new ReluImpl(input, name: name);
}

abstract class Select implements Tensor {
  factory Select(conditionInput, thenInput, elseInput, {String name}) =>
      new SelectImpl(conditionInput, thenInput, elseInput, name: name);
}

abstract class Loss2 implements Tensor {
  factory Loss2(Tensor expected, Tensor input, {String name}) =>
      new Loss2Impl(expected, input, name: name);
}

/*
abstract class SigmoidCrossEntropy implements Tensor {
  factory SigmoidCrossEntropy(expected, input, {String name}) =>
      new SigmoidCrossEntropyImpl(expected, input, name: name);
}
*/
abstract class SigmoidCrossEntropyWithLogits implements Tensor {
  factory SigmoidCrossEntropyWithLogits(expected, logit, {String name}) =>
      new SigmoidCrossEntropyWithLogitsImpl(expected, logit, name: name);
}

abstract class Softmax implements Tensor {
  factory Softmax(input, {String name}) => new SoftmaxImpl(input, name: name);
}

/*
abstract class CrossEntropy implements Tensor {
  factory CrossEntropy(expected, input, {String name}) =>
      new CrossEntropyImpl(expected, input, name: name);
}

abstract class SoftmaxCrossEntropy implements Tensor {
  factory SoftmaxCrossEntropy(expected, input, {String name}) =>
      new SoftmaxCrossEntropyImpl(expected, input, name: name);
}
*/
abstract class SoftmaxCrossEntropyWithLogits implements Tensor {
  factory SoftmaxCrossEntropyWithLogits(expected, logit, {String name}) =>
      new SoftmaxCrossEntropyWithLogitsImpl(expected, logit, name: name);
}

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

abstract class ArgMax implements Tensor {
  factory ArgMax(input, {int axis, String name}) =>
      new ArgMaxImpl(input, axis: axis, name: name);
}
