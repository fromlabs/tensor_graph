// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "../tensor.dart";
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
      : this._trainableVariables = trainableVariables != null
            ? new List.unmodifiable(trainableVariables)
            : null,
        this._checkingRate = checkingRate,
        this._checkingDelta = checkingDelta,
        this._checkingThreshold = checkingThreshold,
        super(inputs: {_targetInputName: target}, name: name, type: type);

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
      if (gradient != null) {
        Variable variable = tensor;

        var importVariable = descriptor.import(variable);

        // TODO migliorare inferenza tipi
        // var assigner = variable.assign(importVariable + gradient * (_learningRateSign * learningRate));
        var assigner = variable.assign(importVariable +
            gradient *
                (new Constant(_learningRateSign * learningRate,
                    dataType: variable.dataType)));

        descriptor.addExecutable(assigner);
      } else {
        throw new ArgumentError(
            "Gradient not calculable on ${getInput(_targetInputName)} by $tensor");
      }
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
