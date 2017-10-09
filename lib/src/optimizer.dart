// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import 'operation.dart';
import "group.dart";
import "variable.dart";

import "impl/optimizer.dart";

abstract class Optimizer implements GroupOperation {
  num get learningRate;

  Iterable<Operation> get initializers;
}

abstract class SgdOptimizer implements Optimizer {
  factory SgdOptimizer(target,
          {List<Variable> trainableVariables,
          num learningRate = 0.01,
          num checkingRate = 0,
          num checkingDelta = 1e-6,
          num checkingThreshold = 1e-3,
          String name}) =>
      new SgdOptimizerImpl(target,
          trainableVariables: trainableVariables,
          learningRate: learningRate,
          checkingRate: checkingRate,
          checkingDelta: checkingDelta,
          checkingThreshold: checkingThreshold,
          name: name);
}

abstract class RmsPropOptimizer implements Optimizer {
  factory RmsPropOptimizer(target,
          {List<Variable> trainableVariables,
          num learningRate = 0.001,
          num decayRate = 0.9,
          num eps = 1e-8,
          num checkingRate = 0,
          num checkingDelta = 1e-6,
          num checkingThreshold = 1e-3,
          String name}) =>
      new RmsPropOptimizerImpl(target,
          trainableVariables: trainableVariables,
          learningRate: learningRate,
          decayRate: decayRate,
          eps: eps,
          checkingRate: checkingRate,
          checkingDelta: checkingDelta,
          checkingThreshold: checkingThreshold,
          name: name);

  num get decayRate;

  num get eps;
}
