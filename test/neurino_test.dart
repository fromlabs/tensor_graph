// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:test/test.dart";

import "package:collection/collection.dart";

import "package:tensor_graph/tensor_graph.dart";

import "package:tensor_graph/src/impl/math.dart";

import "package:tensor_graph/src/impl/core.dart";

import "test_utils.dart";

class MyOperation extends OperationBase {
  MyOperation(x, y, {String name})
      : super({"x": x, "y": y}, name, "MyOperation") {
    registerDefaultOutputProduced();
    registerOutputProduced("ext");
  }

  @override
  void computeOperation(OperationDescriptor descriptor) {
    descriptor.defaultOutputValue =
        descriptor.getInputValue("x") + descriptor.getInputValue("y");

    descriptor.setOutputValue(
        "ext", descriptor.getInputValue("x") * descriptor.getInputValue("y"));
  }

  @override
  void buildGradients(GradientsComputersDescriptor descriptor) {
    descriptor.setDefaultOutputGradient(
        "x",
        (TensorGradientDescriptor descriptor) => descriptor
            .backPropagatedGradientValue
            .reduceSum(
                reductionAxis: calculateReductionBroadcastGradientAxis(
                    descriptor.getInputValue("x").shape,
                    descriptor.getInputValue("y").shape))
            .reshape(
                newDimensions: descriptor.getInputValue("x").shape.dimensions));

    descriptor.setDefaultOutputGradient(
        "y",
        (TensorGradientDescriptor descriptor) => descriptor
            .backPropagatedGradientValue
            .reduceSum(
                reductionAxis: calculateReductionBroadcastGradientAxis(
                    descriptor.getInputValue("y").shape,
                    descriptor.getInputValue("x").shape))
            .reshape(
                newDimensions: descriptor.getInputValue("y").shape.dimensions));

    descriptor.setOutputGradient(
        "ext",
        "x",
        (TensorGradientDescriptor descriptor) => (descriptor
                    .backPropagatedGradientValue *
                descriptor.getInputValue("y"))
            .reduceSum(
                reductionAxis: calculateReductionBroadcastGradientAxis(
                    descriptor.getInputValue("x").shape,
                    descriptor.getInputValue("y").shape))
            .reshape(
                newDimensions: descriptor.getInputValue("x").shape.dimensions));

    descriptor.setOutputGradient(
        "ext",
        "y",
        (TensorGradientDescriptor descriptor) => (descriptor
                    .getInputValue("x") *
                descriptor.backPropagatedGradientValue)
            .reduceSum(
                reductionAxis: calculateReductionBroadcastGradientAxis(
                    descriptor.getInputValue("y").shape,
                    descriptor.getInputValue("x").shape))
            .reshape(
                newDimensions: descriptor.getInputValue("y").shape.dimensions));
  }
}

void main() {
  group('Model Tests', () {
    test('Model Tests - 4', () {
      var model = new Model();

      expect(() => new Constant(0, name: "k0"), throwsStateError);

      model.asDefault((_) {
        new Constant(1, name: "k1");
      });

      model.asDefault((_) {
        new Constant(2, name: "k2");

        new Model().asDefault((_) {
          new Constant(3, name: "k3");
        });

        new Constant(4, name: "k4");
      });

      expect(() => new Constant(5, name: "k5"), throwsStateError);
    });
  });

  group('Session Tests', () {
    test('Session Tests - 5', () {
      var session = new Session(new Model());

      session.asDefault((session) {
        new Constant(1);

        new Session().asDefault((session2) {
          new Constant(1);

          expect(() => session2.asDefault((_) {}), throwsStateError);

          expect(() => session.asDefault((_) {}), throwsStateError);
        });

        new Session(new Model()).asDefault((_) {
          new Constant(1);
        });

        session.model.asDefault((_) {
          new Constant(1);
        });

        new Model().asDefault((_) {
          new Constant(1);
        });
      });

      session.asDefault((_) {});
    });
  });

  group('Constant Tests', () {
    test('Constant Tests - 1', () {
      expect(() => new Constant(1), throwsStateError);

      new Model().asDefault((model) {
        var k = new Constant(1);

        expect(k.id, equals("Constant:default"));
        expect(k.model, equals(model));
        expect(k.operationOutputName, equals(Operation.defaultOutputName));
        expect(k.operation, isNotNull);
        expect(k.operation.id, equals("Constant"));
        expect(k.operation.model, equals(model));
        expect(k.operation.type, equals("Constant"));
        expect(k.operation.defaultOutput, equals(k));
        expect(k.operation.getOutput(Operation.defaultOutputName),
            equals(k.operation.defaultOutput));
        expect(k.consumerIds, isEmpty);
      });
    });

    test('Constant Tests - 2', () {
      var k;

      new Model().asDefault((model) {
        k = new Constant(1);

        new Session().asDefault((session) {
          expect(session.run(k).toValue(), equals(1));
        });
      });

      new Model().asDefault((model) {
        new Session().asDefault((session) {
          expect(() => session.run(k), throwsStateError);
        });
      });
    });

    test('Constant Tests - 3', () {
      new Session(new Model()).asDefault((session) {
        var k = new Constant(1);

        expect(session.run(k).toValue(), equals(1));

        expect(session.run(k, feeds: {k: 2}).toValue(), equals(2));

        expect(session.run(k).toValue(), equals(1));
      });
    });

    test('Constant Tests - 4', () {
      new Session(new Model()).asDefault((session) {
        new Add(1, 2);

        new Add(1, 2, name: "MyAdd");

        new Constant(1, name: "Scope.");

        expect(
            hasIdenticalElements(session.model.operationIds, [
              "Add",
              "Add.input1",
              "Add.input2",
              "MyAdd",
              "MyAdd.input1",
              "MyAdd.input2",
              "Scope.Constant"
            ]),
            isTrue);
      });
    });
  });

  group('Reference Tests', () {
    test('Reference Tests - 1', () {
      new Session(new Model()).asDefault((session) {
        var input = new Placeholder(shapeDimensions: []);

        expect(input.operation.inputNames.isEmpty, isTrue);

        expect(() => session.run(input), throwsStateError);

        expect(session.run(input, feeds: {input: 2}).toScalar(), equals(2));

        expect(() => session.run(input), throwsStateError);

        expect(input.id, equals("Placeholder:default"));

        var input2 = new Placeholder.withDefault(1);

        expect(input2.operation.inputNames.isEmpty, isFalse);

        expect(session.run(input2).toScalar(), equals(1));

        expect(session.run(input2, feeds: {input2: 2}).toScalar(), equals(2));

        expect(session.run(input2).toScalar(), equals(1));

        expect(
            hasIdenticalElements(
                session
                    .runs([input2.operation, input2], feeds: {input2: 2})
                    .values
                    .map((array) => array?.toScalar()),
                [null, 2]),
            isTrue);
      });
    });
  });

  group('Variable Tests', () {
    test('Variable Tests - 1', () {
      new Session(new Model()).asDefault((session) {
        var v = new Variable(1);

        expect(() => session.run(v), throwsStateError);

        expect(session.run(v.initialValue).toScalar(), equals(1));

        expect(session.run(v, feeds: {v: 2}).toScalar(), equals(2));

        expect(session.run(v.initializer), isNull);

        expect(session.run(v).toScalar(), equals(1));

        expect(session.run(v.assign(2)).toScalar(), equals(2));

        expect(session.run(v).toScalar(), equals(2));
      });
    });

    test('Variable Tests - 2', () {
      new Session(new Model()).asDefault((session) {
        var v = new Variable(1);

        expect(() => session.run(v), throwsStateError);

        expect(session.run(v.assign(2)).toScalar(), equals(2));

        expect(session.run(v).toScalar(), equals(2));

        expect(session.run(v.assign(3)).toScalar(), equals(3));

        expect(session.run(v).toScalar(), equals(3));

        expect(session.run(v.assign(v.initialValue)).toScalar(), equals(1));

        expect(session.run(v).toScalar(), equals(1));
      });
    });
  });

  group('Math Tests', () {
    test('Math Tests - 1', () {
      new Session(new Model()).asDefault((session) {
        expect(session.run(new Add(1, 2)).toScalar(), equals(3));

        expect(session.run(new Constant(1) + 2).toScalar(), equals(3));
      });
    });

    test('Math Tests - 2', () {
      new Session(new Model()).asDefault((session) {
        expect(session.run(new Sub(1, 2)).toScalar(), equals(-1));

        expect(session.run(new Constant(1) - 2).toScalar(), equals(-1));

        -(new Constant(1));
      });
    });

    test('Math Tests - 3', () {
      new Session(new Model()).asDefault((session) {
        expect(session.run(new Div(1, 2)).toScalar(), equals(0.5));

        expect(session.run(new Constant(1) / 2).toScalar(), equals(0.5));
      });
    });

    test('Math Tests - 4', () {
      new Session(new Model()).asDefault((session) {
        expect(session.run(new Mul(4, 2)).toScalar(), equals(8));

        expect(session.run(new Constant(4) * 2).toScalar(), equals(8));
      });
    });

    test('Math Tests - 5', () {
      new Session(new Model()).asDefault((session) {
        var x = 1;
        var y;

        expect(() => new Add(x, y), throwsArgumentError);
      });
    });
  });

  group('Group Tests', () {
    test('Group Tests - 1', () {
      new Session(new Model()).asDefault((session) {
        var op = new DefaultGroupTensor(
            {"x": 1, "y": 2},
            (descriptor) =>
                (descriptor.getInput("x") + descriptor.getInput("y")) /
                (descriptor.getInput("x") * descriptor.getInput("y")));

        expect(op.shape.isScalar, isTrue);

        expect(session.run(op).toScalar(), equals(1.5));
      });
    });

    test('Group Tests - 2', () {
      new Session(new Model()).asDefault((session) {
        var op = new DefaultGroupTensor(
            {"x": 1, "y": 2}, (inputs) => new Constant(1));

        expect(session.run(op).toScalar(), equals(1));
      });
    });

    test('Group Tests - 3', () {
      new Session(new Model()).asDefault((session) {
        new DefaultGroupTensor(
            {"x": 1, "y": 2},
            (descriptor) =>
                (descriptor.getInput("x") + descriptor.getInput("y")) /
                (descriptor.getInput("x") * descriptor.getInput("y")));

        new DefaultGroupTensor(
            {"x": 1, "y": 2},
            (descriptor) =>
                (descriptor.getInput("x") + descriptor.getInput("y")) /
                (descriptor.getInput("x") * descriptor.getInput("y")));

        new DefaultGroupTensor(
            {"x": 1, "y": 2},
            (descriptor) =>
                (descriptor.getInput("x") + descriptor.getInput("y")) /
                (descriptor.getInput("x") * descriptor.getInput("y")),
            name: "MyOp");

        new DefaultGroupTensor({"x": 1, "y": 2}, (inputs) => new Constant(1));

        expect(
            hasIdenticalElements(session.model.operationIds, [
              "DefaultGroupTensor",
              "DefaultGroupTensor.x",
              "DefaultGroupTensor.y",
              "DefaultGroupTensor_1",
              "DefaultGroupTensor_1.x",
              "DefaultGroupTensor_1.y",
              "MyOp",
              "MyOp.x",
              "MyOp.y",
              "DefaultGroupTensor_2",
              "DefaultGroupTensor_2.x",
              "DefaultGroupTensor_2.y",
            ]),
            isTrue);
      });
    });

    test('Group Tests - 4', () {
      new Session(new Model()).asDefault((session) {
        new DefaultGroupTensor(
            {"x": 1, "y": 2},
            (descriptor) =>
                (descriptor.getInput("x") + descriptor.getInput("y")) /
                (descriptor.getInput("x") * descriptor.getInput("y")));

        expect(
            hasIdenticalElements(session.model.operationIds, [
              "DefaultGroupTensor",
              "DefaultGroupTensor.x",
              "DefaultGroupTensor.y"
            ]),
            isTrue);

        GroupOperationInternalBase baseGroup =
            session.model.getOperation("DefaultGroupTensor");

        expect(
            hasIdenticalElements(
                baseGroup.operationIds, ["x", "y", "Add", "Mul", "Div"]),
            isTrue);
      });
    });

    test('Group Tests - 5', () {
      new Session(new Model()).asDefault((session) {
        new DefaultGroupTensor({"x": 1, "y": 2}, (inputs) => new Constant(1));

        expect(
            hasIdenticalElements(session.model.operationIds, [
              "DefaultGroupTensor",
              "DefaultGroupTensor.x",
              "DefaultGroupTensor.y"
            ]),
            isTrue);

        GroupOperationInternalBase baseGroup =
            session.model.getOperation("DefaultGroupTensor");

        expect(
            hasIdenticalElements(
                baseGroup.operationIds, ["x", "y", "Constant"]),
            isTrue);
      });
    });

    test('Group Tests - 6', () {
      new Session(new Model()).asDefault((session) {
        var x = 1;
        var y;
        var op = new DefaultGroupTensor(
            {"x": x, "y": y}, (descriptor) => descriptor.getInput("x"));

        expect(hasIdenticalElements(op.operation.inputNames, ["x"]), isTrue);

        expect(session.run(op).toScalar(), equals(1));
      });
    });

    test('Group Tests - 7', () {
      new Session(new Model()).asDefault((session) {
        expect(
            () => new DefaultGroupTensor(
                {"x": null}, (descriptor) => descriptor.getInput("x")),
            throwsArgumentError);

        expect(
            new GroupOperation({"x": null}, (descriptor) {}).hasDefaultOutput,
            isFalse);

        expect(
            new GroupOperation({"x": null}, (descriptor) {
              if (descriptor.hasInput("x")) {
                descriptor.defaultOutput = descriptor.getInput("x");
              }
            }).hasDefaultOutput,
            isFalse);
      });
    });

    test('Group Tests - 8', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(1, name: "x");

        var y = new DefaultGroupTensor({"x": x}, (inputs) => x);

        expect(x.consumerIds.length, equals(1));
        expect(x.consumerIds.contains(y.operation.id), isTrue);
      });
    });
  });
/*
  group('Import Tests', () {
    test('Import Tests - 1', () {
      new Session(new Model()).asDefault((session) {
        var k = new Constant(1);

        var op = new GroupOperation(null, (descriptor) {
          var kInternal = descriptor.import(k);

          descriptor.defaultOutput = kInternal;
        });

        expect(session.run(op.defaultOutput), equals(1));

        var op2 =
            new DefaultGroupTensor(null, (descriptor) => descriptor.import(k));

        expect(session.run(op2), equals(1));
      });
    });

    test('Import Tests - 2', () {
      new Session(new Model()).asDefault((session) {
        var k = new Constant(1);

        Tensor kInternal;
        var op = new GroupOperation({"input": k}, (descriptor) {
          kInternal = new Constant(1);
          expect(() => descriptor.import(kInternal.operation),
              throwsArgumentError);

          expect(() => descriptor.import(descriptor.getInput("input")),
              throwsArgumentError);

          descriptor.defaultOutput = descriptor.getInput("input");
        });

        expect(session.run(session.model.import(kInternal)), equals(1));

        expect(session.run(op.defaultOutput), equals(1));

        var op2 =
            new DefaultGroupTensor(null, (descriptor) => descriptor.import(k));

        expect(session.run(op2), equals(1));
      });
    });
  });
*/
  group('Gradient Tests', () {
    test('Gradient Tests - 1', () {
      new Session(new Model()).asDefault((session) {
        var w0 = new Constant(2, name: "w0");
        var x0 = new Constant(-1, name: "x0");
        var w1 = new Constant(-3, name: "w1");
        var x1 = new Constant(-2, name: "x1");
        var w2 = new Constant(-3, name: "w2");

        var y = new Inv(new Exp(-(w0 * x0 + w1 * x1 + w2)) + 1, name: "y");

        var delta = 0.001;

        var halfDelta = delta / 2;

        var y2 = session.run(y, feeds: {x0: -1 + halfDelta});

        var y1 = session.run(y, feeds: {x0: -1 - halfDelta});

        var dydx = (y2 - y1) / delta;

        expect(dydx.toScalar(), equals(0.3932238547075251));
      });
    });

    test('Gradient Tests - 2', () {
      new Session(new Model()).asDefault((session) {
        var w0 = new Constant(2, name: "w0");
        var x0 = new Placeholder(shapeDimensions: [], name: "x0");
        var w1 = new Constant(-3, name: "w1");
        var x1 = new Placeholder(shapeDimensions: [], name: "x1");
        var w2 = new Constant(-3, name: "w2");

        var y = new Inv(new Exp(-(w0 * x0 + w1 * x1 + w2)) + 1, name: "y");

        var dydx = session.model.gradient(y, [x0]).gradients[x0];

        expect(session.run(dydx, feeds: {x0: -1, x1: -2}).toScalar(),
            0.39322386648296376);
      });
    });

    test('Gradient Tests - 3', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");
        var z = new Constant(-4, name: "z");
        var q = new Add(x, y, name: "q");
        var f = new Mul(q, z, name: "f");

        var gradients =
            session.model.gradient(f, [x, y, z, q]).gradients.values;

        var result = mapMap(session.runs(gradients),
            key: (tensor, value) => tensor.operationOutputName,
            value: (tensor, value) => round(value.toScalar(), 0.0001));

        print(result);

        expect(result.length, equals(4));
        expect(result["dq"], equals(-4));
        expect(result["dx"], equals(-4));
        expect(result["dy"], equals(-4));
        expect(result["dz"], equals(3));
      });
    });

    test('Gradient Tests - 4', () {
      new Session(new Model()).asDefault((session) {
        var c = new Placeholder(shapeDimensions: [], name: "c");
        var x = new Placeholder(shapeDimensions: [], name: "x");
        var y = new Placeholder(shapeDimensions: [], name: "y");
        var z = new Select(c, x, y, name: "z");

        var gradients = session.model.gradient(z, [x, y]).gradients.values;

        var result = mapMap(
            session.runs(gradients, feeds: {c: true, x: 1, y: 0}),
            key: (tensor, value) => tensor.operationOutputName,
            value: (tensor, value) => round(value.toScalar(), 0.0001));

        expect(result.length, equals(2));
        expect(result["dx"], equals(1));
        expect(result["dy"], equals(0));
      });
    });

    test('Gradient Tests - 5', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");
        var z = new Constant(-4, name: "z");
        var q = new Add(x, y, name: "q");
        var r = new Placeholder.withDefault(q, name: "r");
        var f = new Mul(r, z, name: "f");

        var gradients =
            session.model.gradient(f, [x, y, z, q, r]).gradients.values;

        var result = mapMap(session.runs(gradients),
            key: (tensor, value) => tensor.operationOutputName,
            value: (tensor, value) => round(value.toScalar(), 0.0001));

        expect(result.length, equals(5));
        expect(result["dr"], equals(-4));
        expect(result["dq"], equals(-4));
        expect(result["dx"], equals(-4));
        expect(result["dy"], equals(-4));
        expect(result["dz"], equals(3));

        result = mapMap(session.runs(gradients, feeds: {r: 3}),
            key: (tensor, value) => tensor.operationOutputName,
            value: (tensor, value) => round(value.toScalar(), 0.0001));

        expect(result.length, equals(5));
        expect(result["dr"], equals(-4));
        expect(result["dq"], equals(0));
        expect(result["dx"], equals(0));
        expect(result["dy"], equals(0));
        expect(result["dz"], equals(3));
      });
    });

    test('Gradient Tests - 6', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");
        var z = new Constant(-4, name: "z");
        var q = new Add(x, y, name: "q");
        var f = new Mul(q, z, name: "f");

        var gradients =
            session.model.gradient(f, [x, y, z, q]).gradients.values;

        var result = mapMap(session.runs(gradients),
            key: (tensor, value) => tensor.operationOutputName,
            value: (tensor, value) => round(value.toScalar(), 0.0001));

        expect(result.length, equals(4));
        expect(result["dq"], equals(-4));
        expect(result["dx"], equals(-4));
        expect(result["dy"], equals(-4));
        expect(result["dz"], equals(3));

        result = mapMap(session.runs(gradients, feeds: {q: 3}),
            key: (tensor, value) => tensor.operationOutputName,
            value: (tensor, value) => round(value.toScalar(), 0.0001));

        expect(result.length, equals(4));
        expect(result["dq"], equals(-4));
        expect(result["dx"], equals(0));
        expect(result["dy"], equals(0));
        expect(result["dz"], equals(3));
      });
    });

    test('Gradient Tests - 7', () {
      new Session(new Model()).asDefault((session) {
        var thenInput = new Constant(-2, name: "then");
        var elseInput = new Constant(5, name: "else");
        var conditionInput = new Constant(true, name: "condition");
        var select =
            new Select(conditionInput, thenInput, elseInput, name: "select");

        var group = session.model
            .gradient(select, [thenInput, elseInput, conditionInput]);
        var gradients = group.gradients;

        expect(gradients.length, equals(3));
        expect(session.run(gradients[thenInput]).toScalar(), equals(1));
        expect(session.run(gradients[elseInput]).toScalar(), equals(0));
        expect(gradients[conditionInput], isNull);
      });
    });

    test('Gradient Tests - 8', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");
        var z = new Constant(-4, name: "z");
        var q = new Add(x, y, name: "q");
        var f = new Mul(q, z, name: "f");

        var analyticGradients =
            session.model.gradient(f, [q, z, x, y]).gradients.values;

        session.runs(analyticGradients);

        session.runs(analyticGradients, feeds: {q: 2});
      });
    });

    test('Gradient Tests - 9', () {
      new Session(new Model()).asDefault((session) {
        var x = new Constant(-2, name: "x");
        var y = new Constant(5, name: "y");
        var conditionInput = new IsGreaterOrEqual(x, y, name: "condition");
        var select = new Select(conditionInput, x, y, name: "select");

        var analyticGradients =
            session.model.gradient(select, [x, y]).gradients.values;

        session.runs(analyticGradients);

        session.runs(analyticGradients, feeds: {x: 6});

        session.runs(analyticGradients, feeds: {conditionInput: true});
      });
    });
  });

  group('Operation base Tests', () {
    test('Model Tests - 4', () {
      new Session(new Model()).asDefault((session) {
        var op = new MyOperation(1, 2);

        expect(
            hasIdenticalElements(session.model.operationIds,
                ["MyOperation", "MyOperation.x", "MyOperation.y"]),
            isTrue);

        var result = mapMap(
            session.runs([op.defaultOutput, op.getOutput("ext")]),
            key: (tensor, value) => tensor.id);

        expect(result.length, equals(2));
        expect(result["MyOperation:default"].toScalar(), equals(3));
        expect(result["MyOperation:ext"].toScalar(), equals(2));

        var ddefs = session.model.gradient(
            op.defaultOutput, [op.getInput("x"), op.getInput("y")]).gradients;
        var dexts = session.model.gradient(op.getOutput("ext"),
            [op.getInput("x"), op.getInput("y")]).gradients;

        var ddefdx = ddefs[op.getInput("x")];
        var ddefdy = ddefs[op.getInput("y")];
        var dextdx = dexts[op.getInput("x")];
        var dextdy = dexts[op.getInput("y")];

        var gradients = [ddefdx, ddefdy, dextdx, dextdy];

        result = session.runs(gradients);

        expect(result.length, equals(4));
        expect(result[ddefdx].toScalar(), equals(1));
        expect(result[ddefdy].toScalar(), equals(1));
        expect(result[dextdx].toScalar(), equals(2));
        expect(result[dextdy].toScalar(), equals(1));
      });
    });
  });
}
