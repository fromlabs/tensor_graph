// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "group.dart";
import "variable.dart";

import "impl/optimizer.dart";

abstract class Optimizer implements GroupOperation {
  num learningRate;
}

abstract class Minimizer implements Optimizer {
  factory Minimizer(target,
          {List<Variable> trainableVariables,
          num learningRate = 0.01,
          num checkingRate = 0,
          num checkingDelta = 1e-10,
          num checkingThreshold = 1e-3,
          String name}) =>
      new MinimizerImpl(target,
          trainableVariables: trainableVariables,
          learningRate: learningRate,
          checkingRate: checkingRate,
          checkingDelta: checkingDelta,
          checkingThreshold: checkingThreshold,
          name: name);
}

abstract class Maximizer implements Optimizer {
  factory Maximizer(target,
          {List<Variable> trainableVariables,
          num learningRate = 0.01,
          num checkingRate = 0,
          num checkingDelta = 1e-10,
          num checkingThreshold = 1e-3,
          String name}) =>
      new MaximizerImpl(target,
          trainableVariables: trainableVariables,
          learningRate: learningRate,
          checkingRate: checkingRate,
          checkingDelta: checkingDelta,
          checkingThreshold: checkingThreshold,
          name: name);
}
