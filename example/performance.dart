// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:async";
import "dart:math";

import "package:tensor_graph/tensor_graph.dart";
import "package:tensor_math/tensor_math.dart";

import "mnist_generator.dart" as mnist;
import "batch_generator.dart";

var random = new Random();

Future main() async {
  var steps = 100;

  new Session(new Model()).asDefault((session) {
    var x = new ModelInput(shapeDimensions: [1000, 1000], name: "x");
    
    // var x2 = new Transpose(x, permutationAxis: [0, 1]);
    // var x2 = new Transpose(x, permutationAxis: [1, 0]);

    // var y = new ReduceSum(new Inv(new Abs(new Neg(x2))));

    // var y = new ReduceSum(new Neg(new Neg(new Neg(x2))));

    var y = new Neg(x);
    //var y = new Neg2(x);

    var watch = new Stopwatch();
    watch.start();

    print("Start...");

    var data = createRandomData(x.shape.dimensions);

    for (var i in range(1, steps)) {
      session.run(y, feeds: {x: data});
    }

    print("Finish in ${watch.elapsedMilliseconds} ms");
  });
}

NDArray createRandomData(List<int> shapeDimensions) =>
    new NDArray.generate(shapeDimensions, (index) => random.nextDouble());
