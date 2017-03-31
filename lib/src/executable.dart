// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "graph.dart";
import "model.dart";

typedef Executable GraphContextualizedProvider();

abstract class Executable {
  String get id;

  String get path;

  Model get model;

  Graph get graph;

  bool get isExecuted;

  bool get isNotExecuted;

  Iterable<Graph> get importingGraphs;
}
