// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import 'package:test/test.dart';
import "package:collection/collection.dart";

import "package:tensor_graph/tensor_graph.dart";

import "test_utils.dart";

class MyOperation extends OperationBase {
  final Tensor target;

  MyOperation(this.target) : super(null, null, "MyOperation");

  @override
  void computeOperation(OperationDescriptor descriptor) {
    print(target.isExecuted);
    print(target.isFeedValue);
    print(target.isEvaluated);
  }
}

class MyGroup extends GroupOperationBase {
  MyGroup(input) : super({"input": input}, null, "MyOperation");

  @override
  void buildOperation(GroupDescriptor descriptor) {
    print("buildOperation");

    var internalInput = descriptor.getInput("input");

    print(internalInput);
    print(internalInput.operation.type);
  }

  @override
  void computeOperation(OperationDescriptor descriptor) {
    print("computeOperation");

    var input = descriptor.getInput("input");
    print(input);

    print(descriptor.getInputValue("input"));

    super.computeOperation(descriptor);
  }
}

void main() {
  group('Gradient Tests', () {
    test('Gradient Tests - 2', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");
        var z = new Constant(-4, name: "z");
        Tensor q;

        var f = new DefaultGroupTensor({
          "i1": x,
          "i2": y,
          "i3": z,
        }, (DefaultGroupTensorDescriptor descriptor) {
          q = new Add(descriptor.getInput("i1"), descriptor.getInput("i2"),
              name: "q");
          var f = new Mul(q, descriptor.getInput("i3"), name: "f");
          return f;
        }, name: "f");

        // print(session.run(f));

        session.model.gradient(f, [q]);

        logModel(session.model);

        // print(session.runs(df.gradients.values));
      });
    });

    test('Gradient Tests - 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");

        var xy = new Add(x, y, name: "xy");

        var xy2 = new Mul(xy, 2, name: "xy2");

        var xy23 = new Mul(xy2, 3, name: "xy23");

        var f = new Mul(xy23, 5, name: "f");

        print(session.run(f));

        var df = session.model.gradient(f, [xy23, xy2, xy, x, y]);

        logModel(session.model);

        print(session.runs(df.gradients.values));
      });
    });

    test('Gradient Tests - 4', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");
        var xy = new Add(x, y, name: "xy");
        Tensor xy2;

        var xy23 = new DefaultGroupTensor({"i": xy},
            (DefaultGroupTensorDescriptor descriptor) {
          xy2 = new Mul(descriptor.getInput("i"), 2, name: "xy2");
          return new Mul(xy2, 3, name: "output");
        }, name: "xy23");

        new Mul(xy23, 5, name: "f");
      });
    });

    test('Group - 5', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");

        var internalX;
        var internalY;
        var internalOp;

        var op = new DefaultGroupTensor({"i1": x, "i2": y}, (descriptor) {
          internalX = descriptor.getInput("i1");
          internalY = descriptor.getInput("i2");

          internalOp = (internalX + internalY) / (internalX * internalY);

          return internalOp;
        });

        var op2 = op * 2;

        logModel(session.model);

        print("$x consumers: ${x.consumerIds}");
        print("$op consumers: ${op.consumerIds}");
        print("$op2 consumers: ${op2.consumerIds}");
        print("$internalX consumers: ${internalX.consumerIds}");
        print("$internalOp consumers: ${internalOp.consumerIds}");
      });
    });

    test('Gradient Tests - 6', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");
        var xy = new Mul(x, y, name: "xy");

        var gradient = xy.operation.gradient(Operation.defaultOutputName,
            xy.operation.inputNames.toList(), new Constant(1));

        logModel(session.model);

        print(session.run(xy));

        print(
            session.runs([gradient.getOutput("dx"), gradient.getOutput("dy")]));

        print(session.run(xy, feeds: {x: 1}));

        print(session.runs([gradient.getOutput("dx"), gradient.getOutput("dy")],
            feeds: {x: 1}));
      });
    });

    test('Graph Tests - 7', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");
        var z = new Constant(10, name: "z");

        var internalX;
        var internalY;
        var internalOp;

        var op =
            new DefaultGroupTensor({"i1": x, "i2": y, "i3": z}, (descriptor) {
          internalX = descriptor.getInput("i1");
          internalY = descriptor.getInput("i2");
          var internalZ = descriptor.getInput("i3");

          internalOp = (internalX + internalY) / (internalX * internalY);

          return internalOp;
        }, name: "op");

        var op2 = new Mul(op, 2, name: "op2");

        var analyticGradients =
            session.model.gradient(op2, [x, y]).gradients.values;

        logModel(session.model);

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Graph Tests - 8', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");
        new Constant(10, name: "z");

        var op = new Div(x + y, x * y, name: "op");

        var op2 = new Mul(op, 2, name: "op2");

        var analyticGradients =
            session.model.gradient(op2, [x, y]).gradients.values;

        logModel(session.model);

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Gradient Tests - 9', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");
        var z = new Constant(-4, name: "z");
        Tensor q;

        var f = new DefaultGroupTensor({
          "i1": x,
          "i2": y,
          "i3": z,
        }, (DefaultGroupTensorDescriptor descriptor) {
          q = new Add(descriptor.getInput("i1"), descriptor.getInput("i2"),
              name: "q");
          var f = new Mul(q, descriptor.getInput("i3"), name: "f");
          return f;
        }, name: "f");

        var analyticGradients = session.model.gradient(f, [q]).gradients.values;

        logModel(session.model);

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });

    test('Gradient Tests - 10b', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");
        Tensor q;

        var f = new DefaultGroupTensor({
          "i1": x,
          "i2": y,
        }, (descriptor) {
          q = new Add(descriptor.getInput("i1"), descriptor.getInput("i2"),
              name: "q");
          return q;
        }, name: "f");

        var analyticGradients = session.model.gradient(f, [q]).gradients.values;

        logModel(session.model);

        print(session.runs(analyticGradients));
      });
    });

    test('Gradient Tests - 12', () {
      new Session(new Model()).asDefault((session) {
        var a = new Reference(name: "a");

        var b1, b2;
        var b = new DefaultGroupTensor({
          "b1": a,
        }, (descriptor) {
          b1 = descriptor.getInput("b1");
          b2 = new Reference(target: b1, name: "b2");
          return b2;
        }, name: "b");

        var c = new Reference(target: b, name: "c");

        var d1, d2;
        var d = new DefaultGroupTensor({
          "d1": c,
        }, (descriptor) {
          d1 = descriptor.getInput("d1");
          d2 = new Reference(target: d1, name: "d2");
          return d2;
        }, name: "d");

        var e = new Reference(target: d, name: "e");

        logModel(session.model);

        var analyticGradients;

        analyticGradients =
            session.model.gradient(e, <Tensor>[a]).gradients.values;

        print(analyticGradients);

        print(session.runs(analyticGradients, feeds: {a: 1}));

        analyticGradients =
            session.model.gradient(d2, <Tensor>[d1]).gradients.values;

        print(analyticGradients);

        print(session.runs(analyticGradients, feeds: {a: 1}));

        analyticGradients =
            session.model.gradient(e, <Tensor>[b1]).gradients.values;

        print(analyticGradients);

        print(session.runs(analyticGradients, feeds: {a: 1}));

        analyticGradients =
            session.model.gradient(d2, <Tensor>[a]).gradients.values;

        print(analyticGradients);

        print(session.runs(analyticGradients, feeds: {a: 1}));

        analyticGradients =
            session.model.gradient(d2, <Tensor>[b1]).gradients.values;

        print(analyticGradients);

        print(session.runs(analyticGradients, feeds: {a: 1}));
      });
    });

    test('Gradient Tests - 13', () {
      new Session(new Model()).asDefault((session) {
        var a = new Reference(name: "a");

        var a2 = new Reference(target: a, name: "a2");

        var b1, b2;
        var b = new DefaultGroupTensor({
          "b1": a2,
        }, (descriptor) {
          b1 = descriptor.getInput("b1");
          b2 = new Reference(target: b1, name: "b2");
          return b2;
        }, name: "b");

        var c = new Reference(target: b, name: "c");

        logModel(session.model);

        var analyticGradients;

        analyticGradients =
            session.model.gradient(c, <Tensor>[a]).gradients.values;

        print(session.runs(analyticGradients, feeds: {a: 1}));
      });
    });

    test('Gradient Tests - 14', () {
      new Session(new Model()).asDefault((session) {
        var a = new Reference(name: "a");

        var a2 = new Reference(target: a, name: "a2");

        var b = new Reference(target: a2, name: "b");

        var c = new Reference(target: b, name: "c");

        logModel(session.model);

        var analyticGradients;

        analyticGradients =
            session.model.gradient(c, <Tensor>[a]).gradients.values;

        print(session.runs(analyticGradients, feeds: {a: 1}));
      });
    });

    test('Gradient Tests - 15', () {
      var aExpected = 2;

      new Session(new Model()).asDefault((session) {
        var a = new Variable(0.1, name: "a");
        var x = new Reference(name: "x");

        var predicted = new Mul(a, x, name: "predicted");
        var expected = new Reference(name: "expected");

        var loss = new Loss2(expected, predicted, name: "loss");

        session.run(a.initializer);

        var xValue = 1;

        var expectedValue = aExpected * xValue;

        var values = session
            .runs([a, loss], feeds: {x: xValue, expected: expectedValue});

        var aValue = values[a];
        var lossValue = values[loss];

        print("a = $aValue [expected: $aExpected]");
        print("loss = $lossValue");

        // calcolo gradiente analitico

        var analyticGradients =
            session.model.gradient(loss, [a]).gradients.values;

        print("analyticGradients: $analyticGradients");

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients,
                feeds: {x: xValue, expected: expectedValue}),
            key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print("analyticValues: $analyticValues");
      });
    });

    test('Gradient Tests - 16', () {
      var aExpected = 2;

      new Session(new Model()).asDefault((session) {
        var a = new Variable(0.1, name: "a");
        var x = new Reference(name: "x");

        var predicted = new Mul(a, x, name: "predicted");
        var expected = new Reference(name: "expected");

        var loss = new Loss2(expected, predicted, name: "loss");

        session.run(a.initializer);

        var xValue = 1;

        var expectedValue = aExpected * xValue;

        var values = session
            .runs([a, loss], feeds: {x: xValue, expected: expectedValue});

        var aValue = values[a];
        var lossValue = values[loss];

        print("a = $aValue [expected: $aExpected]");
        print("loss = $lossValue");

        // calcolo gradiente analitico

        var analyticGradients = session.model.gradient(loss, [a]).gradients;

        print("analyticGradients: $analyticGradients");

        var analyticValues = session.runs(analyticGradients.values,
            feeds: {x: xValue, expected: expectedValue});

        print("analyticValues: $analyticValues");

        var analyticValuesBySource = mapMap<Tensor, Tensor, Tensor, dynamic>(
            analyticGradients, value: (source, gradient) {
          return analyticValues[gradient];
        });

        print("analyticValuesBySource: $analyticValuesBySource");

        var learningRate = -0.01;

        print("aValue: $aValue");

        aValue += learningRate * analyticValuesBySource[a];

        print("new aValue: $aValue");

        session.run(a.assign(aValue));

        aValue = session.run(a);

        print("aValue: $aValue");
      });
    });

    test('Gradient Tests - 17', () {
      var aExpected = 2;

      new Session(new Model()).asDefault((session) {
        var a = new Variable(0.1, name: "a");
        var x = new Reference(name: "x");

        var predicted = new Mul(a, x, name: "predicted");
        var expected = new Reference(name: "expected");

        var loss = new Loss2(expected, predicted, name: "loss");

        session.run(a.initializer);

        var xValue = 1;

        var expectedValue = aExpected * xValue;

        var values = session
            .runs([a, loss], feeds: {x: xValue, expected: expectedValue});

        var aValue = values[a];
        var lossValue = values[loss];

        print("a = $aValue [expected: $aExpected]");
        print("loss = $lossValue");

        // calcolo gradiente analitico

        var analyticGradients = session.model.gradient(loss, [a]).gradients;

        var aGradient = analyticGradients[a];

        print("aGradient: $aGradient");

        var learningRate = -0.01;

        var aAssigner = a.assign(a + aGradient * learningRate);

        session.runs([aAssigner], feeds: {x: xValue, expected: expectedValue});

        aValue = session.run(a);

        print("new aValue: $aValue");

        logModel(session.model);
      });
    });

    test('Gradient Tests - 18', () {
      var aExpected = 2;

      new Session(new Model()).asDefault((session) {
        var a = new Variable(0.1, name: "a");
        var x = new Reference(name: "x");

        var predicted = new Mul(a, x, name: "predicted");
        var expected = new Reference(name: "expected");

        var loss = new Loss2(expected, predicted, name: "loss");

        var optimizer = new Minimizer(loss,
            trainableVariables: [a], checkingRate: 0.5, name: "optimizer");

        session.run(a.initializer);

        var xValue = 1;

        var expectedValue = aExpected * xValue;

        var values;

        for (var i = 1; i <= 10; i++) {
          print("*** STEP $i ***");
          values = session.runs([a, loss, optimizer],
              feeds: {x: xValue, expected: expectedValue});
          print("loss = ${values[loss]}");
          print("new value: ${session.run(a)} [expected: $aExpected]");
          print("");
          print("");
        }
      });
    });

    test('Graph Tests - 19', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");

        var op = new Mul(x, y, name: "op");

        var analyticGradients =
            session.model.gradient(op, [x, y]).gradients.values;

        var analyticValues = mapMap<Executable, dynamic, String, dynamic>(
            session.runs(analyticGradients), key: (key, value) {
          Tensor tensor = key;
          return tensor.operationOutputName;
        });

        print(analyticValues);
      });
    });
  });
}
