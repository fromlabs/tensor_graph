// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "package:tensor_math/tensor_math.dart" as tm;

import "../tensor.dart";
import "../operation.dart";
import "../group.dart";
import "../variable.dart";
import "../optimizer.dart";
import "../math.dart";

abstract class OptimizerBase extends GroupOperationBase implements Optimizer {
  static const String _targetInputName = "target";

  static const String _initializerSingletonKey = "_INITIALIZER";

  @override
  final num learningRate;

  final num _checkingRate;

  final num _checkingDelta;

  final num _checkingThreshold;

  final List<Variable> _trainableVariables;

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
        _variableModifier(descriptor, tensor, gradient);
      } else {
        throw new ArgumentError(
            "Gradient not calculable on ${getInput(_targetInputName)} by $tensor");
      }
    });
  }

  @override
  Iterable<Operation> get initializers => _trainableVariables
          .where((variable) => hasOperation("cache.${variable.operation.id}"))
          .map((variable) {
        Variable cache =
            getOperation("cache.${variable.operation.id}").defaultOutput;

        return cache.initializer;
      });

  void _variableModifier(
      GroupDescriptor descriptor, Variable inputVariable, Tensor gradient);
}

class SgdOptimizerImpl extends OptimizerBase implements SgdOptimizer {
  static const String __type = "SgdOptimizer";

  SgdOptimizerImpl(target,
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
  void _variableModifier(
      GroupDescriptor descriptor, Variable variable, Tensor gradient) {
    // SGD: x += - learning_rate * dx
    // var assigner = variable.assign(importVariable + gradient * (_learningRateSign * learningRate));

    var importVariable = descriptor.import(variable);

    var newVariableValue = importVariable -
        new Constant(learningRate, dataType: variable.dataType) * gradient;

    var variableAssigner = variable.assign(newVariableValue);

    descriptor.addExecutable(variableAssigner);
  }
}

class RmsPropOptimizerImpl extends OptimizerBase implements RmsPropOptimizer {
  static const String __type = "RmsPropOptimizer";

  @override
  final num decayRate;

  @override
  final num eps;

  RmsPropOptimizerImpl(target,
      {List<Variable> trainableVariables,
      num learningRate,
      num decayRate,
      num eps,
      num checkingRate,
      num checkingDelta,
      num checkingThreshold,
      String name})
      : this.decayRate = decayRate,
        this.eps = eps,
        super(target,
            trainableVariables: trainableVariables,
            learningRate: learningRate,
            checkingRate: checkingRate,
            checkingDelta: checkingDelta,
            checkingThreshold: checkingThreshold,
            name: name,
            type: __type);

  @override
  void _variableModifier(
      GroupDescriptor descriptor, Variable variable, Tensor gradient) {
    // RMSprop
    // cache = decay_rate * cache + (1 - decay_rate) * dx**2
    // x += - learning_rate * dx / (np.sqrt(cache) + eps)

    var importVariable = descriptor.import(variable);

    var cache = new Variable(
        new Constant(
            new tm.NDArray.zeros(variable.shape.dimensions,
                dataType: variable.dataType),
            dataType: variable.dataType),
        dataType: variable.dataType,
        name: "cache.${variable.operation.id}");

    var newCacheValue =
        new Constant(decayRate, dataType: variable.dataType) * cache +
            new Constant(1 - decayRate, dataType: variable.dataType) *
                new Pow(gradient, 2);

    var newVariableValue = importVariable -
        (new Constant(learningRate, dataType: variable.dataType) *
            gradient /
            (new Sqrt(newCacheValue) +
                new Constant(eps, dataType: variable.dataType)));

    var cacheAssigner = cache.assign(newCacheValue);

    var variableAssigner = variable.assign(newVariableValue);

    descriptor.addExecutable(cacheAssigner);
    descriptor.addExecutable(variableAssigner);
  }
}
