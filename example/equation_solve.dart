// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math";

import "package:tensor_graph/tensor_graph.dart";

const int steps = 50000;
const minX = -10;
const maxX = 10;
const num learningRate = 0.0002;

void main() {
  var watch = new Stopwatch();
  watch.start();

  // SOLVE EQUATION
  // y = 5 * x^2 + 3 * x - 2
  const aExpected = 5;
  const bExpected = 3;
  const cExpected = -2;

  var l = 0;
  var random = new Random();

  new Session(new Model()).asDefault((session) {
    var x = new Reference(name: "x");

    var expected = new Reference(
        defaultInput: (new Constant(aExpected) * x * x) +
            (new Constant(bExpected) * x) +
            new Constant(cExpected),
        name: "expected");

    var a = new Variable(0.1, name: "a");
    var b = new Variable(0.1, name: "b");
    var c = new Variable(0, name: "c");

    var predicted = new Reference(
        defaultInput: (a * x * x) + (b * x) + c, name: "predicted");

    var loss = new Loss2(expected, predicted, name: "loss");

    var trainableVariables = [a, b, c];

    var optimizer = new Minimizer(loss,
        trainableVariables: trainableVariables,
        learningRate: learningRate,
        checkingRate: 1,
        name: "optimizer");

    // TODO inizializzazione delle variabili del modello
    session.runs(trainableVariables.map((variable) => variable.initializer));

    var aValue, bValue, cValue, predictedValue, expectedValue, lossValue;

    for (var i = 0; i < steps; i++) {
      var xValue = (random.nextDouble() * (maxX - minX)) + minX;

      var values = session.runs([a, b, c, predicted, expected, loss, optimizer],
          feeds: {x: xValue});

      aValue = values[a];
      bValue = values[b];
      cValue = values[c];
      predictedValue = values[predicted];
      expectedValue = values[expected];
      lossValue = values[loss];

      l++;

      if (i % 1000 == 0) {
        print("******* STEP $i *******");
        print("a = $aValue [expected: $aExpected]");
        print("b = $bValue [expected: $bExpected]");
        print("c = $cValue [expected: $cExpected]");
        print("predicted = $predictedValue [expected: $expectedValue]");
        print("loss = $lossValue");
      }
    }

    print("******* FINAL *******");
    print("a = $aValue [expected: $aExpected]");
    print("b = $bValue [expected: $bExpected]");
    print("c = $cValue [expected: $cExpected]");
    print("predicted = $predictedValue [expected: $expectedValue]");
    print("loss = $lossValue");
  });

  print("$l runs in ${watch.elapsedMilliseconds} ms");
}
