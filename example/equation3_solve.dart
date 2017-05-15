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
  var epochs = 100000;
  var minX = -10;
  var maxX = 10;
  var learningRate = 0.0002;

  new Session(new Model()).asDefault((session) {
    var x = new Placeholder(shapeDimensions: [null, 1], name: "x");
    var expected =
        new Placeholder(shapeDimensions: [null, 1], name: "expected");

    var a = new Variable(0.1, name: "a");
    var b = new Variable(0.1, name: "b");
    var c = new Variable(0, name: "c");

    var predicted = new Named((a * x * x) + (b * x) + c, name: "predicted");

    var loss = new ReduceMean(new Loss2(expected, predicted, name: "loss"));

    var trainableVariables = [a, b, c];

    var optimizer = new Minimizer(loss,
        trainableVariables: trainableVariables,
        learningRate: learningRate,
        checkingRate: 0,
        name: "optimizer");

    // TODO inizializzazione delle variabili del modello
    session.runs(trainableVariables.map((variable) => variable.initializer));

    var dataset = getDataset(datasetCount, minX, maxX);

    for (var i in range(0, epochs)) {
      var values = session.runs([loss, optimizer],
          feeds: {x: dataset["inputs"], expected: dataset["expecteds"]});

      if (i % 10 == 0) {
        print("$i: ${values[loss]}");
      }

      if (values[loss].toScalar() <= 0.001) {
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
  var values = session.runs([
    predicted,
    loss,
  ], feeds: {
    x: dataset["inputs"],
    expected: dataset["expecteds"]
  });

  print("Loss: ${values[loss]}");
}
