// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:collection/collection.dart";

import "package:tensor_graph/tensor_graph.dart";

import "package:tensor_graph/src/impl/core.dart";

num round(num value, double precision) =>
    (value / precision).round() * precision;

bool hasIdenticalElements(Iterable<dynamic> iter1, Iterable<dynamic> iter2) {
  var equality = new UnorderedIterableEquality();
  var equals = equality.equals(iter1, iter2);

  if (!equals) {
    print("${iter1.toList()}\r\n != \r\n${iter2.toList()}");
  }

  return equals;
}

Iterable<Operation> getOperations(Model model) sync* {
  var operationIds = model.operationIds.toList();
  for (var operationId in operationIds) {
    yield model.getOperation(operationId);
  }
}

Iterable<Tensor> getTensors(Model model) sync* {
  for (var operation in getOperations(model)) {
    for (var outputName in operation.outputNames) {
      yield operation.getOutput(outputName);
    }
  }
}

Iterable<Tensor> getTensorContributorsIterator(Tensor target) =>
    _getContributorIterator(target, new Set<Tensor>());

void logModel(Model model) {
  ModelImpl impl = model;
  impl.log();
}

Iterable<Tensor> _getContributorIterator(
    Tensor target, Set<Tensor> collectedContributors) sync* {
  for (var contributorName in target.operation.inputNames) {
    var contributor = target.operation.getInput(contributorName);

    if (!collectedContributors.contains(contributor)) {
      collectedContributors.add(contributor);

      yield contributor;

      yield* _getContributorIterator(contributor, collectedContributors);
    }
  }
}
