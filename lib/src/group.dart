// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "graph.dart";
import "executable.dart";
import "operation.dart";
import "tensor.dart";

import "impl/group.dart";

export "impl/core.dart" show GroupOperationBase;
export "impl/group.dart" show DefaultGroupTensorBase;

typedef dynamic DefaultGroupTensorBuilder(
    DefaultGroupTensorDescriptor descriptor);

typedef void GroupBuilder(GroupDescriptor descriptor);

abstract class GroupOperation implements Operation {
  factory GroupOperation(Map<String, dynamic> inputs, GroupBuilder builder,
          {String name}) =>
      new GroupOperationImpl(inputs, builder, name: name);

  Graph get parent;
}

abstract class DefaultGroupTensor implements Tensor {
  factory DefaultGroupTensor(
          Map<String, dynamic> inputs, DefaultGroupTensorBuilder builder,
          {String name}) =>
      new DefaultGroupTensorImpl(inputs, builder, name: name);
}

abstract class DefaultGroupTensorDescriptor {
  Iterable<String> get inputNames;

  bool hasInput(String name);

  Tensor getInput(String name);

  bool hasImport(Executable executable);

  E import<E extends Executable>(E executable);
}

abstract class GroupDescriptor {
  Iterable<String> get inputNames;

  bool hasInput(String name);

  Tensor getInput(String name);

  bool hasImport(Executable executable);

  E import<E extends Executable>(E executable);

  set defaultOutput(defaultOutput);

  void setOutput(String name, output);

  void addExecutable(Executable executable);
}
