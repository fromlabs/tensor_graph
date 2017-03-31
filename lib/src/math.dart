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

abstract class Equal implements Tensor {
  factory Equal(input1, input2, {String name}) =>
      new EqualImpl(input1, input2, name: name);
}

abstract class NotEqual implements Tensor {
  factory NotEqual(input1, input2, {String name}) =>
      new NotEqualImpl(input1, input2, name: name);
}

abstract class Less implements Tensor {
  factory Less(input1, input2, {String name}) =>
      new LessImpl(input1, input2, name: name);
}

abstract class LessEqual implements Tensor {
  factory LessEqual(input1, input2, {String name}) =>
      new LessEqualImpl(input1, input2, name: name);
}

abstract class Greater implements Tensor {
  factory Greater(input1, input2, {String name}) =>
      new GreaterImpl(input1, input2, name: name);
}

abstract class GreaterEqual implements Tensor {
  factory GreaterEqual(input1, input2, {String name}) =>
      new GreaterEqualImpl(input1, input2, name: name);
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

abstract class BinaryCrossEntropyLoss implements Tensor {
  factory BinaryCrossEntropyLoss(expected, input, {String name}) =>
      new BinaryCrossEntropyLossImpl(expected, input, name: name);
}

abstract class BinaryCrossEntropyWithLogitLoss implements Tensor {
  factory BinaryCrossEntropyWithLogitLoss(expected, logit, {String name}) =>
      new BinaryCrossEntropyWithLogitLossImpl(expected, logit, name: name);
}

abstract class ReduceMean implements Tensor {
  Tensor input;

  factory ReduceMean(input, {String name}) {
    // TODO to implement ReduceMean
    throw new UnimplementedError("to implement ReduceMean");
  }
}
