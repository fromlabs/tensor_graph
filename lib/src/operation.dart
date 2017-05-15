// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:tensor_math/tensor_math.dart";

import "executable.dart";
import "tensor.dart";
import "gradient.dart";

export "impl/core.dart" show OperationBase;

typedef NDShapeable TensorGradientComputer(TensorGradientDescriptor descriptor);

abstract class Operation implements Executable {
  static String defaultOutputName = "default";

  String get type;

  Iterable<String> get inputNames;

  bool hasInput(String name);

  Tensor getInput(String name);

  Iterable<String> get outputNames;

  bool get hasDefaultOutput;

  bool hasOutput(String name);

  Tensor get defaultOutput;

  Tensor getOutput(String name);

  bool isDifferentiable(String outputName, String inputName);

  Differentiator gradient(
      String outputName, List<String> inputNames, backPropagatedGradient,
      {String name});
}

abstract class OperationDescriptor {
  bool get isCalculatingShape;

  NDShapeable toNDShapeable(value);

  Iterable<String> get inputNames;

  bool hasInput(String name);

  Tensor getInput(String name);

  NDShapeable getInputValue(String name);

  set defaultOutputValue(NDShapeable value);

  void setOutputValue(String name, NDShapeable value);
}

abstract class GradientsComputersDescriptor {
  Iterable<String> get inputNames;

  bool hasInput(String name);

  Tensor getInput(String name);

  void setDefaultOutputGradient(
      String inputName, TensorGradientComputer gradientComputer);

  void setOutputGradient(String outputName, String inputName,
      TensorGradientComputer gradientComputer);
}

abstract class TensorGradientDescriptor {
  Iterable<String> get inputNames;

  bool hasInput(String name);

  Tensor getInput(String name);

  NDShapeable getInputValue(String name);

  Tensor get output;

  NDShapeable get outputValue;

  Tensor get backPropagatedGradient;

  NDShapeable get backPropagatedGradientValue;
}
