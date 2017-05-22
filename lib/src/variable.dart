// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:tensor_math/tensor_math.dart";

import "operation.dart";
import "tensor.dart";

import "impl/variable.dart";

abstract class Variable implements Tensor {
  factory Variable(initialValue, {NDDataType dataType, String name}) =>
      new VariableImpl(initialValue, dataType: dataType, name: name);

  Tensor get initialValue;

  Tensor assign(value);

  Operation get initializer;
}
