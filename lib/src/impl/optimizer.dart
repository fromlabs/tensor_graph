// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "../group.dart";
import "../variable.dart";
import "../optimizer.dart";

abstract class OptimizerBase extends GroupOperationBase implements Optimizer {
  static const String _targetInputName = "target";

  @override
  num learningRate;

  num _checkingRate;

  num _checkingDelta;

  num _checkingThreshold;

  List<Variable> _trainableVariables;

  OptimizerBase(target,
      {List<Variable> trainableVariables,
      this.learningRate,
      num checkingRate,
      num checkingDelta,
      num checkingThreshold,
      String name,
      String type})
      : this._trainableVariables = trainableVariables,
        this._checkingRate = checkingRate,
        this._checkingDelta = checkingDelta,
        this._checkingThreshold = checkingThreshold,
        super({_targetInputName: target}, name, type);

  num get _learningRateSign;

  @override
  void buildOperation(GroupDescriptor descriptor) {
    var analyticGradients = model
        .gradient(getInput(_targetInputName), _trainableVariables,
            checkingRate: _checkingRate,
            checkingDelta: _checkingDelta,
            checkingThreshold: _checkingThreshold)
        .gradients;

    analyticGradients.forEach((tensor, gradient) {
      Variable variable = tensor;

      var importVariable = descriptor.import(variable);

      var assigner = variable.assign(
          importVariable + gradient * (_learningRateSign * learningRate));

      descriptor.addExecutable(assigner);
    });
  }
}

class MinimizerImpl extends OptimizerBase implements Minimizer {
  static const String __type = "Minimizer";

  MinimizerImpl(target,
      {List<Variable> trainableVariables,
      num learningRate,
      num checkingRate,
      num checkingDelta,
      num checkingThreshold,
      String name})
      : super(target,
            trainableVariables: trainableVariables,
            learningRate: learningRate,
            checkingRate: checkingRate,
            checkingDelta: checkingDelta,
            checkingThreshold: checkingThreshold,
            name: name,
            type: __type);

  @override
  num get _learningRateSign => -1;
}

class MaximizerImpl extends OptimizerBase implements Maximizer {
  static const String __type = "Maximizer";

  MaximizerImpl(target,
      {List<Variable> trainableVariables,
      num learningRate,
      num checkingRate,
      num checkingDelta,
      num checkingThreshold,
      String name})
      : super(target,
            trainableVariables: trainableVariables,
            learningRate: learningRate,
            checkingRate: checkingRate,
            checkingDelta: checkingDelta,
            checkingThreshold: checkingThreshold,
            name: name,
            type: __type);

  @override
  num get _learningRateSign => 1;
}
