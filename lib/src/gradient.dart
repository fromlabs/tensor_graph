// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "operation.dart";
import "tensor.dart";

abstract class Differentiator implements Operation {
  Map<Tensor, Tensor> get gradients;
}
