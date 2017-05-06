// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "graph.dart";
import "executable.dart";
import "operation.dart";
import "tensor.dart";
import "gradient.dart";

import "impl/core.dart";

abstract class Model extends Graph {
  factory Model() => new ModelImpl();

  void asDefault(void scopedRunnable(Model model));

  Iterable<String> get operationIds;

  bool hasOperation(String id);

  Operation getOperation(String id);

  bool hasTensor(String id);

  Tensor getTensor(String id);

  bool hasImport(Executable executable);

  E import<E extends Executable>(E executable);

  bool isDifferentiable(Tensor target, Tensor source);

  Differentiator gradient(Tensor target, List<Tensor> sources,
      {num checkingRate = 0,
      num checkingDelta = 1e-10,
      num checkingThreshold = 1e-3,
      String name});

  Differentiator numericGradient(Tensor target, List<Tensor> sources,
      {num delta = 1e-10, String name});
}
