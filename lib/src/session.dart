// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:tensor_math/tensor_math.dart";

import "model.dart";
import "executable.dart";
import "tensor.dart";

import "impl/core.dart";

abstract class Session {
  factory Session([Model model]) => new SessionImpl(model);

  Model get model;

  bool get isClosed;

  void close();

  void asDefault(void scopedRunnable(Session session));

  NDArray run(Executable target, {Map<Tensor, dynamic> feeds});

  Map<Executable, NDArray> runs(Iterable<Executable> targets,
      {Map<Tensor, dynamic> feeds});
}

abstract class ExecutableState {
  bool get isExecuted;

  bool get isNotExecuted;

  bool contains(String key);

  void remove(String key);

  dynamic operator [](String key);

  void operator []=(String key, value);

  bool containsInSession(String key);

  void removeFromSession(String key);

  dynamic getFromSession(String key);

  void setInSession(String key, value);
}

abstract class OperationState implements ExecutableState {}

abstract class TensorState implements ExecutableState {
  bool get isEvaluated;

  bool get isNotEvaluated;

  bool get isExecutionValue;

  bool get isFeedValue;
}
