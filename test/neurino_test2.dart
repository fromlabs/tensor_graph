// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:test/test.dart";

import "package:tensor_graph/tensor_graph.dart";

import "package:tensor_math/tensor_math.dart";

class MyTensor extends DefaultDifferentiableTensorBase {
  static const String __type = "MyTensor";

  static const String _inputInputName = "input";

  MyTensor(input, {String name})
      : super({_inputInputName: input}, name, __type);

  @override
  NDShapeable computeValue(DefaultTensorDescriptor descriptor) {
    if (descriptor.isCalculatingShape) {
      return descriptor.getInputValue(_inputInputName);
    } else {
      return new NDArray([1, 2]);
    }
  }

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(
        _inputInputName,
        (TensorGradientDescriptor descriptor) =>
            descriptor.backPropagatedGradientValue);
  }
}

class MyTensor2 extends DefaultDifferentiableTensorBase {
  static const String __type = "MyTensor2";

  static const String _inputInputName = "input";

  MyTensor2(input, {String name})
      : super({_inputInputName: input}, name, __type);

  @override
  NDShapeable computeValue(DefaultTensorDescriptor descriptor) =>
      descriptor.getInputValue(_inputInputName);

  @override
  void buildDefaultGradients(OutputGradientComputersDescriptor descriptor) {
    descriptor.setOutputGradient(_inputInputName,
        (TensorGradientDescriptor descriptor) {
          descriptor.getInputValue(_inputInputName);

          return new NDArray([1, 2]);
        });
  }
}

void main() {
  group('Model Tests', () {
    test('Model Tests - 4', () {
      new Session(new Model()).asDefault((session) {
        var input = new ModelInput(shapeDimensions: [2]);

        print(session.run(input, feeds: {
          input: [1]
        }));
      });
    });

    test('Model Tests - 5', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(1);

        var y = new MyTensor(x);

        print(session.run(y));
      });
    });

    test('Model Tests - 6', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(1);

        var y = new MyTensor(x);

        var gradients = session.model.gradient(y, [x]).gradients;

        print(gradients[x].shape);

        print(session.run(gradients[x]));
      });
    });

    test('Model Tests - 7', () {
      new Session(new Model()).asDefault((session) {
        var x = new ModelInput(shapeDimensions: [null]);

        var y = new MyTensor2(x);

        var gradients = session.model.gradient(y, [x]).gradients;

        print(gradients[x].shape);

        print(session.run(gradients[x], feeds: {x: [1, 2, 3]}));
      });
    });
  });
}
