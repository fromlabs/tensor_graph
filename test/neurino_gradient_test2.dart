// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import 'package:test/test.dart';
import "package:collection/collection.dart";

import "package:tensor_graph/tensor_graph.dart";

import "test_utils.dart";

void main() {
  group('Gradient Tests', () {
    test('Graph Tests - add 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-2, 3], name: "x");
        var y = new Constant([5, 5], name: "y");

        var op = new Add(x, y, name: "op");

        print(session.run(op));

        var analyticGradients = session.model
            .gradient(op, [x, y], checkingRate: 1)
            .gradients
            .values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - add 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-2, 3], name: "x");
        var y = new Constant(5, name: "y");

        var op = new Add(x, y, name: "op");

        print(session.run(op));

        var analyticGradients = session.model
            .gradient(op, [x, y], checkingRate: 1)
            .gradients
            .values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - add 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [-2, 3],
          [1, 4]
        ], name: "x");
        var y = new Constant([5, 1], name: "y");

        var op = new Add(x, y, name: "op");

        print(session.run(op));

        var analyticGradients = session.model
            .gradient(op, [x, y], checkingRate: 1)
            .gradients
            .values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - sub', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");

        var op = new Sub(x, y, name: "op");

        var analyticGradients = session.model
            .gradient(op, [x, y], checkingRate: 1)
            .gradients
            .values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - mul', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");

        var op = new Mul(x, y, name: "op");

        var analyticGradients = session.model
            .gradient(op, [x, y], checkingRate: 1)
            .gradients
            .values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - mul 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [1, 2],
          [3, 4],
          [5, 6]
        ], name: "x");
        var y = new Constant([
          [1],
          [2]
        ], name: "y");

        var z = new MatMul(x, y, name: "z");

        print(session.run(z));

        var analyticGradients =
            session.model.gradient(z, [x, y], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - matmul', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [1, 2, 3],
          [4, 5, 6]
        ], name: "x");
        var y = new Constant([
          [1],
          [2],
          [3]
        ], name: "y");

        var op = new MatMul(x, y, name: "op");

        var analyticGradients = session.model
            .gradient(op, [x, y], checkingRate: 1)
            .gradients
            .values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - matmul2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [
            [1, 2, 3],
            [4, 5, 6]
          ]
        ], name: "x");
        var y = new Constant([
          [
            [1],
            [2],
            [3]
          ]
        ], name: "y");

        var op = new MatMul(x, y, name: "op");

        var analyticGradients = session.model
            .gradient(op, [x, y], checkingRate: 1)
            .gradients
            .values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - div 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");

        var op = new Div(x, y, name: "op");

        var analyticGradients = session.model
            .gradient(op, [x, y], checkingRate: 1)
            .gradients
            .values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - div 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-5, 10], name: "x");
        var y = new Constant(5, name: "y");

        var op = new Div(x, y, name: "op");

        print(session.run(op));

        var analyticGradients = session.model
            .gradient(op, [x, y], checkingRate: 1)
            .gradients
            .values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - div 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [-5, 10],
          [5, 4]
        ], name: "x");
        var y = new Constant([5, 2], name: "y");

        var op = new Div(x, y, name: "op");

        print(session.run(op));

        var analyticGradients = session.model
            .gradient(op, [x, y], checkingRate: 1)
            .gradients
            .values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - inv 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");

        var op = new Inv(x, name: "op");

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - inv 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-5, 10], name: "x");

        var op = new Inv(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - inv 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [-5, 10],
          [5, 4]
        ], name: "x");

        var op = new Inv(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - neg 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");

        var op = new Neg(x, name: "op");

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - neg 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-5, 10], name: "x");

        var op = new Neg(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - neg 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [-5, 10],
          [5, 4]
        ], name: "x");

        var op = new Neg(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - abs 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");

        var op = new Abs(x, name: "op");

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Abs 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-5, 10], name: "x");

        var op = new Abs(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Abs 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [-5, 10],
          [5, 4]
        ], name: "x");
        var op = new Abs(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Exp 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");

        var op = new Exp(x, name: "op");

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Exp 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-5, 10], name: "x");

        var op = new Exp(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Exp 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [-5, 10],
          [5, 4]
        ], name: "x");
        var op = new Exp(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Log 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");

        var op = new Log(x, name: "op");

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Log 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-5, 10], name: "x");

        var op = new Log(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Log 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [-5, 10],
          [5, 4]
        ], name: "x");
        var op = new Log(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Sigmoid 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");

        var op = new Sigmoid(x, name: "op");

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Sigmoid 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-5, 10], name: "x");

        var op = new Sigmoid(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Sigmoid 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [-5, 10],
          [5, 4]
        ], name: "x");
        var op = new Sigmoid(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Tanh 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");

        var op = new Tanh(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Tanh 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-5, 10], name: "x");

        var op = new Tanh(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Tanh 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [-5, 10],
          [5, 4]
        ], name: "x");
        var op = new Tanh(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Relu 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");

        var op = new Relu(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Relu 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-5, 10], name: "x");

        var op = new Relu(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Relu 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [-5, 10],
          [5, 4]
        ], name: "x");
        var op = new Relu(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Select 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(true, name: "x");
        var y = new Constant(1, name: "y");
        var z = new Constant(-1, name: "z");

        var op = new Select(x, y, z, name: "op");

        print(session.run(op));

        var analyticGradients = session.model
            .gradient(op, [y, z], checkingRate: 0)
            .gradients
            .values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Select 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(true, name: "x");
        var y = new Constant([-5, 10], name: "y");
        var z = new Constant([5, -10], name: "z");

        var op = new Select(x, y, z, name: "op");

        print(session.run(op));

        var analyticGradients = session.model
            .gradient(op, [y, z], checkingRate: 1)
            .gradients
            .values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - Select 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(true, name: "x");
        var y = new Constant([
          [-5, 10],
          [5, 4]
        ], name: "y");
        var z = new Constant([
          [5, -10],
          [-5, -4]
        ], name: "z");

        var op = new Select(x, y, z, name: "op");

        print(session.run(op));

        var analyticGradients = session.model
            .gradient(op, [y, z], checkingRate: 1)
            .gradients
            .values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - ReduceSum 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");

        var op = new ReduceSum(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - ReduceSum 1b', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");

        var op = new ReduceSum(x, name: "op");

        print(session.run(op));

        var numericGradients =
            session.model.numericGradient(op, [x]).gradients.values;

        var numericValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(numericGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(numericValues);
      });
    });

    test('Graph Tests - ReduceSum 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-5, 10], name: "x");

        var op = new ReduceSum(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - ReduceSum 2b', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-5, 10], name: "x");

        var op = new ReduceSum(x, name: "op");

        print(session.run(op));

        var numericGradients =
            session.model.numericGradient(op, [x]).gradients.values;

        var numericValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(numericGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(numericValues);
      });
    });

    test('Graph Tests - ReduceSum 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [-5, 10],
          [5, 4]
        ], name: "x");

        var op = new ReduceSum(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - ReduceSum 3b', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [-5, 10],
          [5, 4]
        ], name: "x");

        var op = new ReduceSum(x, name: "op");

        print(session.run(op));

        var numericGradients =
            session.model.numericGradient(op, [x]).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(numericGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - ReduceMean 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");

        var op = new ReduceMean(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - ReduceMean 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([-5, 10], name: "x");

        var op = new ReduceMean(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - ReduceMean 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [-5, 10],
          [5, 4]
        ], name: "x");

        var op = new ReduceMean(x, name: "op");

        print(session.run(op));

        var analyticGradients =
            session.model.gradient(op, [x], checkingRate: 1).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - xor solve 1', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1]
        ]);
        var expected = new Constant([0, 1, 1, 0]);

        var wl1 = new Constant([
          [1, 1],
          [1, 1]
        ], name: "w_l1");
        var bl1 = new Constant([0, 0], name: "b_l1");

        var wl2 = new Constant([
          [1],
          [1]
        ], name: "w_l2");
        var bl2 = new Constant(0, name: "b_l2");

        var logitl1 =
            new Reference(target: new MatMul(x, wl1) + bl1, name: "logit_l1");

        var outputl1 = new Sigmoid(logitl1, name: "output_l1");

        var logitl2 = new Reference(
            target: new MatMul(outputl1, wl2) + bl2, name: "logit_l2");

        // TODO media

        var loss = new SigmoidCrossEntropyWithLogitLoss(expected, logitl2);

        var f = loss;

        print("loss: ${session.run(f)}");

        var gradients = session.model.numericGradient(f, [x]).gradients.values;

        print(session.runs(gradients));
      });
    });

    test('Graph Tests - xor solve 2', () {
      new Session(new Model()).asDefault((session) {
        // 4 x 2
        var x = new Constant([
          [0, 0],
          [0, 1],
          [1, 0],
          [1, 1]
        ]);

        // 2 x 2
        var wl1 = new Constant([
          [1, 1],
          [1, 1]
        ], name: "w_l1");

        // 4 x 2
        var logitl1 = new MatMul(x, wl1);

        // 2 x 1
        var wl2 = new Constant([
          [1],
          [1]
        ], name: "w_l2");

        // 4 x 1
        var logitl2 = new MatMul(logitl1, wl2);

        // 1 x 4
        var expected = new Constant([
          [0, 1, 1, 0]
        ]);

        // 4 x 4
        var loss = expected + logitl2;

        print(session.runs([expected, logitl2, loss]));

        var gradients =
            session.model.gradient(loss, [logitl1]).gradients.values;
/*
        var numericGradients =
            session.model.numericGradient(loss, [logitl1]).gradients.values;

        print(session.runs(numericGradients));
*/
        print(session.runs(gradients));
      });
    });

    test('Graph Tests - xor solve 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant([
          [0, 1, 1, 0]
        ], name: "x");
        var y = new Constant([
          [0],
          [1],
          [1],
          [0]
        ], name: "y");

        var z = x + y;

        print("x: ${session.run(x)}");
        print("y: ${session.run(y)}");
        print("z: ${session.run(z)}");

        print(session.runs(session.model.gradient(z, [x, y]).gradients.values));
      });
    });

    test('Graph Tests - xor solve 4', () {
      new Session(new Model()).asDefault((session) {
        var x = new Reference(shape: [null, 2], name: "x");
        var expected = new Reference(shape: [null, 1], name: "expected");

        var wl1 = new Variable([
          [1, 1],
          [1, 1]
        ], name: "w_l1");
        var bl1 = new Variable([0, 0], name: "b_l1");

        var wl2 = new Variable([
          [1],
          [1]
        ], name: "w_l2");
        var bl2 = new Variable(0, name: "b_l2");

        var trainableVariables = [wl1, bl1, wl2, bl2];

        var logitl1 =
            new Reference(target: new MatMul(x, wl1) + bl1, name: "logit_l1");

        var outputl1 = new Sigmoid(logitl1, name: "output_l1");

        var logitl2 = new Reference(
            target: new MatMul(outputl1, wl2) + bl2, name: "logit_l2");

        var loss = new SigmoidCrossEntropyWithLogitLoss(expected, logitl2);

        session
            .runs(trainableVariables.map((variable) => variable.initializer));

        print(session.runs([
          expected,
          logitl2,
          loss
        ], feeds: {
          x: [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
          ],
          expected: [0, 1, 1, 0]
        }));

        var gradients = session.model
            .gradient(loss, [outputl1], checkingRate: 1)
            .gradients
            .values;

        print(session.runs(gradients, feeds: {
          x: [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
          ],
          expected: [0, 1, 1, 0]
        }));
      });
    });
  });
}
