// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:test/test.dart";

import "package:collection/collection.dart";

import "package:tensor_graph/tensor_graph.dart";

import "package:tensor_graph/src/impl/math.dart";

import "package:tensor_graph/src/impl/core.dart";

import "test_utils.dart";

void main() {
  group('Model Tests', () {
    test('Model Tests - 4', () {
      new Session(new Model()).asDefault((session) {
        var input = new Placeholder(shapeDimensions: [null]);

        print(session.run(input, feeds: {input: [1]}));
      });
    });
  });
}
