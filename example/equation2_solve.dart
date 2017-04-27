// Copyright (c) 2016, Roberto Tassi. All rights reserved. Use of this source code
// is governed by a BSD-style license that can be found in the LICENSE file.

import "dart:math";

import "package:tensor_graph/tensor_graph.dart";

const ka = 5;
const kb = 3;
const kc = -2;

var random = new Random();

num nextDouble(num from, num to) => random.nextDouble() * (to - from) + from;

num function(num x) {
  return ka * x * x + kb * x + kc;
}

Map<String, dynamic> getDataset(int count, num minX, num maxX) {
  var inputs = [];
  var expecteds = [];
  for (var _ in range(0, count)) {
    var x = nextDouble(minX, maxX);
    inputs.add(x);
    expecteds.add(function(x));
  }
  return {"inputs": inputs, "expecteds": expecteds};
}

void main() {
  var watch = new Stopwatch();
  watch.start();

  var datasetCount = 1000;
  var epochs = 100;
  var minX = -10;
  var maxX = 10;
  var learningRate = 0.0002;

  new Session(new Model()).asDefault((session) {
    var x = new Reference(name: "x");
    var expected = new Reference(name: "expected");

    var a = new Variable(0.1, name: "a");
    var b = new Variable(0.1, name: "b");
    var c = new Variable(0, name: "c");

    var predicted =
        new Reference(target: (a * x * x) + (b * x) + c, name: "predicted");

    var loss = new Loss2(expected, predicted, name: "loss");

    var trainableVariables = [a, b, c];

    var optimizer = new Minimizer(loss,
        trainableVariables: trainableVariables,
        learningRate: learningRate,
        checkingRate: 1,
        name: "optimizer");

    // TODO inizializzazione delle variabili del modello
    session.runs(trainableVariables.map((variable) => variable.initializer));

    var dataset = getDataset(datasetCount, minX, maxX);

    for (var i in range(0, epochs)) {
      var lossAverage = 0.0;
      for (var entry in enumerate(dataset["inputs"])) {
        var l = entry.index;
        var inputValue = entry.element;
        var expectedValue = dataset["expecteds"][l];

        var values = session.runs([loss, optimizer],
            feeds: {x: inputValue, expected: expectedValue});

        lossAverage += values[loss];
      }
      lossAverage /= dataset["inputs"].length;

      if (i % 10 == 0) {
        print("$i: $lossAverage");
      }

      if (lossAverage <= 0.01) {
        break;
      }

      if (i % 1000 == 0) {
        test(dataset, x, predicted, expected, loss, session);
      }
    }

    test(dataset, x, predicted, expected, loss, session);

    var values = session.runs([a, b, c]);

    print("a = ${values[a]} [expected: $ka]");
    print("b = ${values[b]} [expected: $kb]");
    print("c = ${values[c]} [expected: $kc]");
  });
}

void test(Map<String, dynamic> dataset, Tensor x, Tensor predicted,
    Tensor expected, Tensor loss, Session session) {
  var lossAverage = 0.0;
  for (var entry in enumerate(dataset["inputs"])) {
    var i = entry.index;
    var inputValue = entry.element;
    var expectedValue = dataset["expecteds"][i];

    var values = session.runs([
      predicted,
      loss,
    ], feeds: {
      x: inputValue,
      expected: expectedValue
    });

    lossAverage += values[loss];

    print("Input: $inputValue");
    print("Expected: $expectedValue");
    print("Predicted: ${values[predicted] >= 0.5} [${values[predicted]}]");
  }
  lossAverage /= dataset["inputs"].length;

  print("Loss: $lossAverage");
}
