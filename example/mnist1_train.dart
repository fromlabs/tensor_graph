// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:async";
import "dart:io";
import "dart:convert";
import "dart:math";

import "package:tensor_graph/tensor_graph.dart";
import "package:tensor_math/tensor_math.dart";

var random = new Random();

Future<Map<String, dynamic>> getDataset() async {
  var json = new File("dataset/mnist/dataset.json");
  return JSON.decode(json.readAsStringSync());
}

Future main() async {
  var watch = new Stopwatch();
  watch.start();

  var dataset = await getDataset();

  var trainDataset = dataset["train"];
  var testDataset = dataset["test"];

  var steps = 1000;
  var batchSize = 100;
  var learningRate = 0.00000005;

  var generator =
      new BatchGenerator(trainDataset["images"].length, new Random(0));

  new Session(new Model()).asDefault((session) {
    var x = new Reference(shape: [null, 784], name: "x");
    var w = new Variable(new NDArray.zeros([784, 10]));
    var b = new Variable(new NDArray.zeros([10]));
    var y = new MatMul(x, w) + b;

    var expected = new Reference(shape: [null, 10], name: "expected");

    // var sm = new Softmax(y);

    var loss = new ReduceMean(new SoftmaxCrossEntropyWithLogits(expected, y));

    var trainableVariables = [w, b];

    var optimizer = new Minimizer(loss,
        trainableVariables: trainableVariables,
        learningRate: learningRate,
        name: "optimizer");

    var correctPrediction = new IsEqual(new ArgMax(y, axis: 1), new ArgMax(expected, axis: 1));

    var accuracy = new ReduceMean(new Select(correctPrediction, 1, 0));

    // TODO inizializzazione delle variabili del modello
    session.runs(trainableVariables.map((variable) => variable.initializer));

    for (var i in range(0, steps)) {
      var indexes = generator.getBatchIndexes(batchSize);

      var imagesBatch =
          extractBatchFromIndexes(trainDataset["images"], indexes);
      var labelsBatch =
          extractBatchFromIndexes(trainDataset["labels"], indexes);

      var values = session.runs([loss, optimizer],
          feeds: {x: imagesBatch, expected: labelsBatch});

      if (i % 1 == 0) {
        print("$i: ${values[loss]}");
      }
    }

    test(testDataset, x, y, expected, loss, correctPrediction, accuracy, session);
  });
}

void test(Map<String, dynamic> dataset, Tensor x, Tensor y, Tensor expected,
    Tensor loss, Tensor correctPrediction, Tensor accuracy, Session session) {
  print("*** TEST ***");

  var values = session.runs([loss, correctPrediction, accuracy],
      feeds: {x: dataset["images"], expected: dataset["labels"]});

  print("Loss: ${values[loss]}");
  print("Correct Prediction: ${values[correctPrediction]}");
  print("Accuracy: ${values[accuracy]}");
}

Future<List<int>> readLabels(String path) async {
  var file = new File(path);

  var result = [];
  await for (var data in file.openRead()) {
    for (int i = 0; i < data.length; i++) {
      result.add(data[i]);
    }
  }

  if (result[0] != 0x00 ||
      result[1] != 0x00 ||
      result[2] != 0x08 ||
      result[3] != 0x01) {
    throw new ArgumentError("Wrong magic number");
  }

  var count =
      (result[4] << 24) + (result[5] << 16) + (result[6] << 8) + result[7];

  if (count != file.lengthSync() - 8) {
    throw new ArgumentError("Invalid length");
  }

  var labels = [];

  var i = 0;
  var offset = 8;
  while (i < count) {
    labels.add(result[offset]);

    offset++;
    i++;
  }

  return labels;
}

Future<List<List<int>>> readImages(String path) async {
  var file = new File(path);

  var result = [];
  await for (var data in file.openRead()) {
    for (int i = 0; i < data.length; i++) {
      result.add(data[i]);
    }
  }

  if (result[0] != 0x00 ||
      result[1] != 0x00 ||
      result[2] != 0x08 ||
      result[3] != 0x03) {
    throw new ArgumentError("Wrong magic number");
  }

  var count =
      (result[4] << 24) + (result[5] << 16) + (result[6] << 8) + result[7];

  var rows =
      (result[8] << 24) + (result[9] << 16) + (result[10] << 8) + result[11];

  var columns =
      (result[12] << 24) + (result[13] << 16) + (result[14] << 8) + result[15];

  if (count * rows * columns != file.lengthSync() - 16) {
    throw new ArgumentError("Invalid length");
  }

  if (rows != 28) {
    throw new ArgumentError("Invalid rows");
  }

  if (columns != 28) {
    throw new ArgumentError("Invalid columns");
  }

  List<List<int>> images = [];

  var i = 0;
  var offset = 16;
  while (i < count) {
    List<int> image = [];
    for (var i2 = 0; i2 < rows * columns; i2++) {
      image.add(result[offset]);

      offset++;
    }

    images.add(image);
    i++;
  }

  return images;
}

List<E> extractBatchFromIndexes<E>(List<E> data, List<int> indexes) =>
    indexes.map<E>((index) => data[index]).toList();

class BatchGenerator {
  final int _count;
  final Random _random;

  int _offset;
  List<int> _data;

  BatchGenerator(this._count, [Random random])
      : _random = random ?? new Random();

  List<int> getBatchIndexes(int batchSize) {
    if (_data == null) {
      _offset = 0;
      _data = new List.generate(_count, (index) => index);
      _data.shuffle(_random);
    }

    var batch;
    var end = _offset + batchSize;
    if (end < _data.length) {
      batch = _data.sublist(_offset, end);
      _offset = end;
    } else {
      batch = [];
      batch.addAll(_data.sublist(_offset));
      var leftCount = end - _data.length;
      _data = null;
      batch.addAll(getBatchIndexes(leftCount));
    }
    return batch;
  }
}
