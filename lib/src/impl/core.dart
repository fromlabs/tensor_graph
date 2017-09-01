// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:async";

import "package:meta/meta.dart";
import "package:collection/collection.dart";

import "package:tensor_math/tensor_math.dart";

import "../graph.dart";
import "../model.dart";
import "../executable.dart";
import "../operation.dart";
import "../tensor.dart";
import "../group.dart";
import "../gradient.dart";
import "../session.dart";
import "../math.dart";
import "../utils.dart";

const String _defaultGraphKey = "_DEFAULT_GRAPH";

const String _defaultSessionKey = "_DEFAULT_SESSION";

const String _executedStateKey = "_EXECUTED";

GraphBase get _defaultGraph =>
    Zone.current[_defaultGraphKey] ??
    (throw new StateError("Default model not available"));

ModelImpl get _defaultModel => _defaultGraph._model;

SessionImpl get _defaultSession =>
    Zone.current[_defaultSessionKey] ??
    (throw new StateError("Default session not available"));

abstract class GraphBase implements Graph {
  static const _defaultType = "Operation";

  final Map<String, int> _autoOperationIds = {};

  final Map<String, OperationInternalBase> _operations = {};

  final Map<OperationInternalBase, _ImportOperationImpl> __imports = {};

  final Set<OperationInternalBase> __exports = new Set();

  @override
  Iterable<String> get importingIds =>
      __imports.values.map((importing) => importing.id);

  @override
  Iterable<String> get exportedIds => __exports.map((exported) => exported.id);

  GraphBase get _parent;

  ModelImpl get _model;

  String get _path;

  Iterable<String> get _operationIds => _operations.keys;

  bool _hasOperation(String id) => _operations.containsKey(id);

  OperationInternalBase _getOperation(String id) =>
      _operations[id] ??
      (throw new ArgumentError("Operation $id not found in $this"));

  bool _hasTensor(String id) {
    var i = id.lastIndexOf(":");
    return i != -1 &&
        _getOperation(id.substring(0, i)).hasOutput(id.substring(i + 1));
  }

  TensorInternalBase _getTensor(String id) {
    var i = id.lastIndexOf(":");
    return i != -1
        ? _getOperation(id.substring(0, i)).getOutput(id.substring(i + 1))
        : (throw new ArgumentError("Tensor $id not found in $this"));
  }

  void _registerOperation(OperationInternalBase operation) {
    _operations[operation.id] = operation;
  }

  bool _hasImport(Executable executable) {
    if (executable is Operation) {
      Operation operation = executable;

      OperationInternalBase baseOperation = operation;

      return __imports.containsKey(baseOperation);
    } else if (executable is Tensor) {
      return _hasImport(executable.operation);
    } else {
      throw new StateError("Not valid executable $executable");
    }
  }

  E _import<E extends Executable>(E executable) {
    if (executable is Operation) {
      Operation operation = executable;
      OperationInternalBase baseOperation = operation;

      if (baseOperation is _ImportOperationImpl) {
        throw new UnsupportedError(
            "Can't import an imported operation $operation");
      }

      dynamic importOperation = __imports.putIfAbsent(baseOperation, () {
        var internal;
        _asDefaultInternal((_) {
          internal = new _ImportOperationImpl(baseOperation);
        });

        return internal;
      });

      return importOperation;
    } else if (executable is Tensor) {
      Tensor tensor = executable;

      dynamic importTensor =
          _import(tensor.operation).getOutput(tensor.operationOutputName);

      return importTensor;
    } else {
      throw new StateError("Not valid executable $executable");
    }
  }

  void _registerOperationExported(OperationInternalBase exportedOperation) {
    __exports.add(exportedOperation);
  }

  @protected
  void _asDefaultInternal(void scopedRunnable(Graph graph)) {
    runZoned(() => scopedRunnable(this), zoneValues: {_defaultGraphKey: this});
  }

  Map<Executable, NDArray> _executes(Iterable<ExecutableBase> targets) =>
      new Map<Executable, NDArray>.fromIterable(targets,
          value: (target) => target._execute());

  bool _isDifferentiable(Tensor target, Tensor source) =>
      _getDifferentiablePathTensors(source, target).isNotEmpty;

  Operation _analyticGradient(
          Tensor target, List<Tensor> sources, backPropagatedGradient,
          {num checkingRate = 0,
          num checkingDelta = 1e-6,
          num checkingThreshold = 1e-3,
          String name}) =>
      new _AnalyticDifferentiatorImpl(target, sources, backPropagatedGradient,
          checkingRate: checkingRate,
          checkingDelta: checkingDelta,
          checkingThreshold: checkingThreshold,
          name: name);

  Operation _numericGradient(Tensor target, List<Tensor> sources,
          {num delta = 1e-6, String name}) =>
      new _NumericDifferentiatorImpl(target, sources, delta: delta, name: name);

  String _nextOperationId(String name, String type) {
    var operationName;
    if (name != null) {
      if (name.endsWith(".")) {
        operationName = "$name${type ?? _defaultType}";
      } else {
        operationName = name;
      }
    } else {
      operationName = type ?? _defaultType;
    }

    var count = (_autoOperationIds[operationName] ?? -1) + 1;
    _autoOperationIds[operationName] = count;

    return count > 0 ? "${operationName}_$count" : operationName;
  }

  void _logGraph([String indentation = ""]) {
    if (exportedIds.isNotEmpty) {
      print("$indentation- exports: ${exportedIds.join(", ")}");
    }

    if (importingIds.isNotEmpty) {
      print("$indentation- imports: ${importingIds.join(", ")}");
    }
  }

  void _logOperations([String indentation = ""]) {
    for (var id in _operationIds) {
      _getOperation(id)._logOperation(indentation);
    }
  }
}

class ModelImpl extends GraphBase implements Model {
  static const String _toString = "<>";

  @override
  final String _path = "";

  @override
  final GraphBase _parent = null;

  @override
  Iterable<String> get operationIds => _operationIds;

  @override
  bool hasOperation(String id) => _hasOperation(id);

  @override
  Operation getOperation(String id) => _getOperation(id);

  @override
  bool hasTensor(String id) => _hasTensor(id);

  @override
  Tensor getTensor(String id) => _getTensor(id);

  @override
  bool hasImport(Executable executable) => _hasImport(executable);

  @override
  E import<E extends Executable>(E executable) => _import(executable);

  @override
  void asDefault(void scopedRunnable(Model model)) {
    _asDefaultInternal((graph) => scopedRunnable(this));
  }

  @override
  bool isDifferentiable(Tensor target, Tensor source) =>
      _isDifferentiable(target, source);

  @override
  Differentiator gradient(Tensor target, List<Tensor> sources,
          {num checkingRate = 0,
          num checkingDelta = 1e-6,
          num checkingThreshold = 1e-3,
          String name}) =>
      _analyticGradient(target, sources, null,
          checkingRate: checkingRate,
          checkingDelta: checkingDelta,
          checkingThreshold: checkingThreshold,
          name: name);

  @override
  Differentiator numericGradient(Tensor target, List<Tensor> sources,
          {num delta = 1e-6, String name}) =>
      _numericGradient(target, sources, delta: delta, name: name);

  @override
  String toString() => _toString;

  void log([String indentation = ""]) {
    print("${indentation}Model $this graph");

    _logGraph(indentation);

    print("$indentation- operations:");
    _logOperations("$indentation  ");
  }

  @override
  ModelImpl get _model => this;
}

abstract class ExecutableBase implements Executable {
  Map<String, Executable> _providers = {};

  dynamic _execute();

  @protected
  Executable provideGraphContextualizedSingleton(
          String singletonKey, GraphContextualizedProvider provide) =>
      _providers.putIfAbsent(
          singletonKey, () => provideGraphContextualized(provide));

  @protected
  Executable provideGraphContextualized(GraphContextualizedProvider provide) {
    var executableGraph = graph;
    if (executableGraph is ModelImpl) {
      Executable executable;
      executableGraph.asDefault((model) {
        executable = provide();
      });
      return executable;
    } else {
      throw new UnsupportedError("$executableGraph contextualization");
    }
  }

  @override
  String toString() => "<$path>";

  @protected
  void _checkModel() {
    if (model != _defaultModel) {
      throw new StateError("Default model not compatible with $this");
    }
  }
}

abstract class OperationInternalBase extends ExecutableBase
    implements Operation {
  @override
  final String id;

  @override
  final String type;

  final GraphBase _graph;

  final Map<String, TensorInternalBase> _inputs = {};

  final Map<String, TensorInternalBase> _outputs = {};

  Map<String, NDDescriptor> _outputDescriptors;

  final Set<_ImportOperationImpl> _importingOperations = new Set();

  OperationInternalBase(Map<String, dynamic> inputs, String name, String type)
      : this.id = _defaultGraph?._nextOperationId(name, type),
        this.type = type,
        this._graph = _defaultGraph {
    _graph._registerOperation(this);

    if (inputs != null) {
      for (var entry in entries(inputs)) {
        if (entry.value != null) {
          _registerInputConsumed(entry.key, entry.value);
        }
      }
    }
  }

  @protected
  void registerDefaultOutputProduced([Tensor output]) {
    registerOutputProduced(Operation.defaultOutputName, output);
  }

  @protected
  void registerOutputProduced(String name, [Tensor output]) {
    TensorInternalBase baseTensor = output ?? new _OperationTensorImpl();

    _outputs[name] = baseTensor;

    baseTensor._registerProducerOperation(this, name);
  }

  @override
  Graph get graph => _graph;

  @override
  Iterable<Graph> get importingGraphs =>
      _importingOperations.map((importing) => importing.graph).toSet();

  @override
  bool get isExecuted => state?.isExecuted ?? false;

  @override
  bool get isNotExecuted => state?.isNotExecuted ?? true;

  @override
  String get path {
    String basePath = _graph._path;
    return basePath.isNotEmpty ? "$basePath/$id" : id;
  }

  @override
  Model get model {
    if (_graph is ModelImpl) {
      ModelImpl model = _graph;
      return model;
    } else {
      return _graph._model;
    }
  }

  @protected
  NDArray run(Executable target, {Map<Tensor, dynamic> feeds}) =>
      _defaultSession.run(target, feeds: feeds);

  @protected
  Map<Executable, NDArray> runs(Iterable<Executable> targets,
          {Map<Tensor, dynamic> feeds}) =>
      _defaultSession.runs(targets, feeds: feeds);

  @override
  Iterable<String> get inputNames => _inputs.keys;

  @override
  Iterable<String> get outputNames => _outputs.keys;

  @override
  bool hasInput(String name) => _inputs.containsKey(name);

  @override
  bool get hasDefaultOutput => hasOutput(Operation.defaultOutputName);

  @override
  bool hasOutput(String name) => _outputs.containsKey(name);

  @override
  Tensor getInput(String name) =>
      _inputs[name] ??
      (throw new ArgumentError("Input $name not registered in $this"));

  @override
  Tensor get defaultOutput => getOutput(Operation.defaultOutputName);

  @override
  Tensor getOutput(String name) =>
      _outputs[name] ??
      (throw new ArgumentError("Output $name not registered in $this"));

  @protected
  OperationState get state => _defaultSession?._getOperationState(this);

  TensorState getInputState(String name) =>
      _defaultSession?._getTensorState(getInput(name));

  TensorState getOutputState(String name) =>
      _defaultSession?._getTensorState(getOutput(name));

  @protected
  void computeOperation(OperationDescriptor descriptor);

  @override
  Differentiator gradient(String outputTargetName,
          List<String> inputSourceNames, backPropagatedGradient,
          {String name}) =>
      _gradient(
          getOutput(outputTargetName),
          inputSourceNames,
          new Map.fromIterable(this.inputNames,
              value: (inputName) => getInput(inputName)),
          backPropagatedGradient,
          name);

  TensorInternalBase _registerInputConsumed(String name, input) {
    var tensor;
    if (input == null || input is TensorInternalBase) {
      tensor = input;
    } else {
      tensor = new Constant(input, name: "$id.$name");
    }

    if (tensor.graph != _graph) {
      throw new StateError("$this and $tensor are not siblings");
    }

    _inputs[name] = tensor;

    tensor._registerConsumerOperation(this, name);

    return tensor;
  }

  void _registerOperationExported(_ImportOperationImpl importingOperation) {
    _importingOperations.add(importingOperation);

    _graph._registerOperationExported(this);
  }

  NDDescriptor _evaluateOutputDescriptor(String name) {
    if (_outputDescriptors == null) {
      _outputDescriptors = {};

      computeOperation(
          new _OperationDescriptorImpl(this, isEvaluatingDescriptor: true));
    }

    getOutput(name);

    return _outputDescriptors[name];
  }

  NDArray _evaluateOutput(String name) {
    _TensorStateImpl tensorState = getOutputState(name);

    if (!tensorState.isFeedValue) {
      _execute();
    }

    return tensorState.value;
  }

  @override
  void _execute() {
    _checkModel();

    if (state.isNotExecuted) {
      _OperationStateImpl stateImpl = state;

      computeOperation(
          new _OperationDescriptorImpl(this, isEvaluatingDescriptor: false));

      stateImpl.setExecuted();
    }
  }

  @protected
  NDDescriptor _getInputDescriptor(String name) {
    TensorInternalBase baseTensor = getInput(name);
    return baseTensor.descriptor;
  }

  @protected
  NDArray _getInputValue(String name) {
    TensorInternalBase baseTensor = getInput(name);
    return baseTensor._execute();
  }

  @protected
  void _setOutputDescriptor(String name, NDDescriptor descriptor) {
    if (descriptor != null) {
      _outputDescriptors[name] = descriptor;
    } else {
      throw new ArgumentError.notNull("Tensor ${getOutput(name)} descriptor");
    }
  }

  @protected
  void _setOutputValue(String name, NDArray value) {
    if (value != null) {
      TensorInternalBase tensor = getOutput(name);

      if (!tensor.descriptor.isCompatibleWith(value.descriptor)) {
        throw new ArgumentError(
            "Computed output value descriptor ${value.descriptor} doesn't match ${tensor.descriptor} in $tensor");
      }

      _TensorStateImpl tensorState = getOutputState(name);

      if (!tensorState.isFeedValue) {
        tensorState.value = value;
      }
    } else {
      throw new ArgumentError.notNull("Tensor ${getOutput(name)} value");
    }
  }

  Differentiator _gradient(Tensor outputTarget, List<String> inputSourceNames,
      Map<String, Tensor> inputs, backPropagatedGradient, String name);

  void _logOperation([String indentation = ""]) {
    print("$indentation- $id [$type:$runtimeType]");
    if (inputNames.isNotEmpty) {
      for (var name in inputNames) {
        _logOperationInput(name, indentation);
      }
    }
    if (outputNames.isNotEmpty) {
      for (var name in outputNames) {
        _logOperationOutput(name, indentation);
      }
    }
  }

  void _logOperationInput(String inputName, [String indentation = ""]) {
    var input = getInput(inputName);
    print(
        "$indentation  > $inputName = ${input.id} [${input.runtimeType}:${input.operation.type}:${input.operation.runtimeType}]");
  }

  void _logOperationOutput(String outputName, [String indentation = ""]) {
    var output = getOutput(outputName);
    print(
        "$indentation  < $outputName = ${output.id} [${output.operation.type}:${output.runtimeType}]");
  }
}

abstract class OperationBase extends OperationInternalBase {
  _GradientsComputersDescriptorImpl __gradientsComputersDescriptor;

  OperationBase(
      {Map<String, dynamic> inputs, String name, @required String type})
      : super(inputs, name, type);

  @override
  bool isDifferentiable(String outputName, String inputName) =>
      _gradientsComputersDescriptor._hasGradientComputer(outputName, inputName);

  @protected
  void buildGradients(GradientsComputersDescriptor descriptor) {}

  @override
  Differentiator _gradient(Tensor outputTarget, List<String> inputSourceNames,
          Map<String, Tensor> inputs, backPropagatedGradient, String name) =>
      new _GradientOperationImpl(
          outputTarget,
          inputSourceNames,
          inputs,
          backPropagatedGradient,
          _gradientsComputersDescriptor
              ._getComputerGradients(outputTarget.operationOutputName),
          name);

  _GradientsComputersDescriptorImpl get _gradientsComputersDescriptor {
    if (__gradientsComputersDescriptor == null) {
      __gradientsComputersDescriptor =
          new _GradientsComputersDescriptorImpl(this);

      buildGradients(__gradientsComputersDescriptor);
    }
    return __gradientsComputersDescriptor;
  }
}

abstract class TensorInternalBase extends ExecutableBase implements Tensor {
  NDDescriptor _descriptor;

  Operation _operation;

  String _operationOutputName;

  final Map<String, OperationInternalBase> _consumers = {};

  TensorInternalBase(NDDataType dataType, [List<int> shapeDimensions])
      : this._descriptor = new NDDescriptor(
            shape: new NDShape(shapeDimensions), dataType: dataType);

  @override
  NDDescriptor get descriptor => _descriptor;

  @override
  NDDataType get dataType => descriptor.dataType;

  @override
  NDShape get shape => descriptor.shape;

  @override
  void setShapeDimensions(List<int> newDimensions) {
    _descriptor = descriptor
        .mergeWith(new NDDescriptor(shape: new NDShape(newDimensions)));
  }

  @override
  String get id => "${operation.id}:$operationOutputName";

  @override
  String get path => "${_internalOperation.path}:$operationOutputName";

  @override
  Model get model => operation.model;

  @override
  Graph get graph => _internalOperation.graph;

  @override
  Operation get operation => _operation;

  @override
  bool get isDefaultOutput =>
      _operationOutputName == Operation.defaultOutputName;

  @override
  String get operationOutputName => _operationOutputName;

  @override
  Iterable<String> get consumerIds => _consumers.keys;

  @override
  Iterable<Graph> get importingGraphs => _operation.importingGraphs;

  @override
  bool get isExecuted => state.isExecuted;

  @override
  bool get isNotExecuted => state.isNotExecuted;

  @override
  bool get isEvaluated => state.isEvaluated;

  @override
  bool get isNotEvaluated => state.isNotEvaluated;

  @override
  bool get isExecutionValue => state.isExecutionValue;

  @override
  bool get isFeedValue => state.isFeedValue;

  TensorState get state =>
      _internalOperation.getOutputState(operationOutputName);

  @override
  Tensor operator +(value) => new Add(this, value);

  @override
  Tensor operator -(value) => new Sub(this, value);

  @override
  Tensor operator -() => new Neg(this);

  @override
  Tensor operator *(value) => new Mul(this, value);

  @override
  Tensor operator /(value) => new Div(this, value);

  @override
  Tensor operator >(value) => new IsGreater(this, value);

  @override
  Tensor operator >=(value) => new IsGreaterOrEqual(this, value);

  @override
  Tensor operator <(value) => new IsLess(this, value);

  @override
  Tensor operator <=(value) => new IsLessOrEqual(this, value);

  @protected
  bool hasInput(String name) => _internalOperation.hasInput(name);

  @protected
  Tensor getInput(String name) => _internalOperation.getInput(name);

  @override
  bool isDifferentiable(String inputName) =>
      _internalOperation.isDifferentiable(operationOutputName, inputName);

  @protected
  NDArray run(Executable target, {Map<Tensor, dynamic> feeds}) =>
      _internalOperation.run(target, feeds: feeds);

  @protected
  Map<Executable, NDArray> runs(Iterable<Executable> targets,
          {Map<Tensor, dynamic> feeds}) =>
      _internalOperation.runs(targets, feeds: feeds);

  OperationInternalBase get _internalOperation => _operation;

  void _registerProducerOperation(
      OperationInternalBase operation, String operationOutputName) {
    _operation = operation;
    _operationOutputName = operationOutputName;

    _descriptor =
        descriptor.mergeWith(_evaluateDescriptor() ?? new NDDescriptor());
  }

  void _registerConsumerOperation(
      OperationInternalBase operation, String operationInputName) {
    _consumers[operation.id] = operation;
  }

  NDDescriptor _evaluateDescriptor() =>
      _internalOperation._evaluateOutputDescriptor(operationOutputName);

  @override
  NDArray _execute() => _internalOperation._evaluateOutput(operationOutputName);
}

abstract class TensorBase extends TensorInternalBase {
  TensorBase({NDDataType dataType}) : super(dataType);
}

abstract class GroupOperationInternalBase extends OperationInternalBase
    with GraphBase {
  final Map<String, TensorInternalBase> _internalInputs = {};

  final Map<String, TensorInternalBase> _internalOutputs = {};

  final List<ExecutableBase> _internalExecutables = [];

  @override
  final GraphBase _parent;

  GroupOperationInternalBase(
      Map<String, dynamic> inputs, String name, String type)
      : this._parent = _defaultGraph,
        super(inputs, name, type) {
    _asDefaultInternal((_) {
      for (var inputName in inputNames) {
        _internalInputs[inputName] =
            new _GroupInternalInputTensorImpl(inputName);
      }

      var descriptor = new _GroupTensorsDescriptorImpl(this, _internalInputs);

      buildOperation(descriptor);

      _internalExecutables.addAll(descriptor._internalExecutables);

      for (var entry in entries(descriptor._internalOutputs)) {
        if (entry.value != null) {
          if (entry.value is Tensor) {
            _internalOutputs[entry.key] = entry.value;
          } else {
            Tensor k = new Constant(entry.value,
                name: entry.key == Operation.defaultOutputName
                    ? "DefaultOutput."
                    : "Output.");
            _internalOutputs[entry.key] = k;
          }
        }
      }
    });

    for (var outputName in _internalOutputs.keys) {
      registerOutputProduced(outputName, createExternalOutput(outputName));
    }
  }

  @protected
  Tensor createExternalOutput(String outputName) =>
      new _GroupExternalOutputTensorImpl(_internalOutputs[outputName]);

  @protected
  void buildOperation(GroupDescriptor descriptor);

  Graph get parent => _parent;

  Iterable<String> get operationIds => _operationIds;

  bool hasOperation(String id) => _hasOperation(id);

  Operation getOperation(String id) => _getOperation(id);

  bool hasTensor(String id) => _hasTensor(id);

  Tensor getTensor(String id) => _getTensor(id);

  bool hasImport(Executable executable) => _hasImport(executable);

  E import<E extends Executable>(E executable) => _import(executable);

  @override
  void computeOperation(OperationDescriptor descriptor) {
    for (var outputName in outputNames) {
      descriptor.setOutputValue(
          outputName,
          descriptor.isEvaluatingDescriptor
              ? _getInternalOutputDescriptor(outputName)
              : _getInternalOutputValue(outputName));
    }

    if (!descriptor.isEvaluatingDescriptor) {
      for (var executable in _internalExecutables) {
        executable._execute();
      }
    }
  }

  @override
  bool isDifferentiable(String outputName, String inputName) =>
      _isDifferentiable(
          _internalOutputs[outputName], _internalInputs[inputName]);

  @override
  String toString() => "<$_path>";

  void log([String indentation = ""]) {
    print("${indentation}Group $this graph");

    _logGraph(indentation);

    print("$indentation- operations:");
    _logOperations("$indentation  ");
  }

  @override
  ModelImpl get _model => _parent._model;

  @override
  String get _path => path;

  Tensor _getInternalInput(String name) => _internalInputs[name];

  Tensor _getInternalOutput(String name) => _internalOutputs[name];

  NDDescriptor _getInternalOutputDescriptor(String name) {
    TensorInternalBase output = _internalOutputs[name];

    return output.descriptor;
  }

  NDArray _getInternalOutputValue(String name) {
    TensorInternalBase internalOutput = _internalOutputs[name];

    return internalOutput._execute();
  }

  @override
  Differentiator _gradient(Tensor outputTarget, List<String> inputSourceNames,
      Map<String, Tensor> inputs, backPropagatedGradient, String name) {
    var internalOutput = _internalOutputs[outputTarget.operationOutputName];
    var internalInputSources = inputSourceNames
        .map((inputSourceName) => _internalInputs[inputSourceName])
        .toList();

    return _analyticGradient(
        internalOutput, internalInputSources, backPropagatedGradient,
        name: name);
  }

  @override
  void _logOperation([String indentation = ""]) {
    super._logOperation(indentation);

    log("$indentation  ");
  }
}

abstract class GroupOperationBase extends GroupOperationInternalBase {
  GroupOperationBase(
      {@required String type, Map<String, dynamic> inputs, String name})
      : super(inputs, name, type);
}

class SessionImpl implements Session {
  final ModelImpl _model;

  final _SessionState _state = new _SessionState();

  SessionImpl([Model model]) : this._model = model ?? _defaultModel;

  @override
  Model get model => _model;

  @override
  bool get isClosed => _state._isClosed;

  @override
  void close() {
    _checkClosed();

    _state._close();
  }

  @override
  void asDefault(void scopedRunnable(Session session)) {
    _checkClosed();

    _checkReentrantSession();

    _model.asDefault((model) => runZoned(() => scopedRunnable(this),
        zoneValues: {_defaultSessionKey: this, this: true}));
  }

  @override
  NDArray run(Executable target, {Map<Tensor, dynamic> feeds}) =>
      runs([target], feeds: feeds)[target];

  @override
  Map<Executable, NDArray> runs(Iterable<Executable> targets,
      {Map<Tensor, dynamic> feeds}) {
    _checkClosed();

    var previousExecutionState = Zone.current[_executedStateKey];
    return runZoned(() => _internalModel._executes(targets), zoneValues: {
      _executedStateKey:
          new _ExecutionState(_state, feeds ?? {}, previousExecutionState)
    });
  }

  ModelImpl get _internalModel => model;

  _ExecutionState get _executionState => Zone.current[_executedStateKey];

  _OperationStateImpl _getOperationState(Operation operation) =>
      _executionState?._getState(operation);

  _TensorStateImpl _getTensorState(Tensor tensor) =>
      _executionState?._getState(tensor);

  void _checkClosed() {
    if (_state._isClosed) {
      throw new StateError("Session is closed");
    }
  }

  void _checkReentrantSession() {
    if (Zone.current[this] != null) {
      throw new StateError("Sessions are not reentrant");
    }
  }
}

class _OperationTensorImpl extends TensorInternalBase {
  _OperationTensorImpl() : super(null);
}

class _GroupInternalInputTensorImpl extends DefaultTensorBase {
  static const String __type = "GroupInternalInput";

  _GroupInternalInputTensorImpl(String name)
      : super(operationName: name, type: __type);

  @override
  NDObject computeValue(DefaultTensorDescriptor descriptor) {
    GroupOperationInternalBase group = graph;

    if (descriptor.isEvaluatingDescriptor) {
      return group._getInputDescriptor(operation.id);
    } else {
      return group._getInputValue(operation.id);
    }
  }

  Tensor get _externalInput {
    GroupOperationInternalBase group = graph;

    return group.getInput(operation.id);
  }
}

class _GroupExternalOutputTensorImpl extends TensorInternalBase {
  Tensor _internalOutput;

  _GroupExternalOutputTensorImpl(this._internalOutput) : super(null);

  @override
  NDDescriptor get descriptor => _internalOutput.descriptor;

  @override
  void setShapeDimensions([List<int> newDimensions]) {
    throw new UnsupportedError("Shape of imported tensor is readonly");
  }
}

class _ImportOperationImpl extends OperationInternalBase {
  static const String __type = "Import";

  final OperationInternalBase _importedOperation;

  _ImportOperationImpl(this._importedOperation)
      : super(null, "#${_getFlatPath(_importedOperation)}", __type) {
    for (var outputName in outputNames) {
      registerOutputProduced(outputName,
          new _ImportTensorImpl(_importedOperation.getOutput(outputName)));
    }
  }

  @override
  Iterable<Graph> get importingGraphs => _importedOperation.importingGraphs;

  @override
  Iterable<String> get inputNames => _importedOperation.inputNames;

  @override
  Iterable<String> get outputNames => _importedOperation.outputNames;

  @override
  bool hasInput(String name) => _importedOperation.hasInput(name);

  @override
  bool hasOutput(String name) => _importedOperation.hasOutput(name);

  @override
  Tensor getInput(String name) {
    var importedTensor = _importedOperation.getInput(name);

    if (_graph._hasImport(importedTensor.operation)) {
      return _graph
          ._import(importedTensor.operation)
          .getOutput(importedTensor.operationOutputName);
    } else {
      throw new ArgumentError.value(
          "Operation ${importedTensor.operation} not imported in $_graph");
    }
  }

  @override
  void computeOperation(OperationDescriptor descriptor) {
    if (!descriptor.isEvaluatingDescriptor) {
      _importedOperation._execute();
    }
  }

  @override
  bool isDifferentiable(String outputName, String inputName) =>
      _importedOperation.isDifferentiable(outputName, inputName);

  @override
  Differentiator _gradient(Tensor outputTarget, List<String> inputSourceNames,
          Map<String, Tensor> inputs, backPropagatedGradient, String name) =>
      _importedOperation._gradient(
          outputTarget, inputSourceNames, inputs, backPropagatedGradient, name);

  @override
  void _logOperation([String indentation = ""]) {
    print(
        "$indentation- $id = import $_importedOperation [$type:$runtimeType]");
  }
}

class _ImportTensorImpl extends TensorInternalBase {
  final TensorInternalBase _importedTensor;

  _ImportTensorImpl(this._importedTensor) : super(null);

  @override
  NDDescriptor get descriptor => _importedTensor.descriptor;

  @override
  void setShapeDimensions([List<int> newDimensions]) {
    throw new UnsupportedError("Shape of imported tensor is readonly");
  }
}

class _GradientOperationImpl extends OperationInternalBase
    implements Differentiator {
  static const String __type = "Gradient";

  static const String _outputInputName = "output";
  static const String _backPropagatedGradientInputName =
      "backPropagatedGradient";

  final List<String> _inputSourceNames;

  final Map<String, String> _gradientInputNames;

  final Map<String, TensorGradientComputer> _gradientsComputers;

  factory _GradientOperationImpl(
      Tensor outputTarget,
      List<String> inputSourceNames,
      Map<String, Tensor> inputs,
      backPropagatedGradient,
      Map<String, TensorGradientComputer> gradientsComputers,
      String name) {
    var allInputs = new Map<String, dynamic>.from(inputs);

    allInputs[_outputInputName] = outputTarget;
    allInputs[_backPropagatedGradientInputName] = backPropagatedGradient;

    return new _GradientOperationImpl._(
        inputSourceNames,
        allInputs,
        gradientsComputers,
        name ?? _getDifferentialName(outputTarget),
        new Map.fromIterable(
            inputSourceNames.where((inputSourceName) =>
                gradientsComputers.containsKey(inputSourceName)),
            value: (sourceName) => _getDifferentialName(inputs[sourceName])));
  }

  _GradientOperationImpl._(this._inputSourceNames, Map<String, dynamic> inputs,
      this._gradientsComputers, String name, this._gradientInputNames)
      : super(inputs, name, __type) {
    for (var gradientInputName in _gradientInputNames.values) {
      registerOutputProduced(gradientInputName);
    }
  }

  @override
  Map<Tensor, Tensor> get gradients => new Map.fromIterable(
          _inputSourceNames.map((inputSourceName) => getInput(inputSourceName)),
          value: (inputSource) {
        var name = _getDifferentialName(inputSource);
        return hasOutput(name) ? getOutput(name) : null;
      });

  @override
  bool isDifferentiable(String outputName, String inputName) {
    return false;
  }

  @override
  void computeOperation(OperationDescriptor descriptor) {
    var tensorDescriptor = new _TensorGradientDescriptorImpl(this, descriptor);

    for (var entry in entries(_gradientInputNames)) {
      if (descriptor.isEvaluatingDescriptor) {
        descriptor.setOutputValue(
            entry.value, descriptor.getInputValue(entry.key));
      } else {
        var tensor = descriptor.getInputValue(entry.key);

        var value = _gradientsComputers[entry.key](tensorDescriptor);

        if (!tensor.descriptor.isCompatibleWith(value.descriptor)) {
          throw new ArgumentError(
              "Computed gradient value descriptor ${value.descriptor} doesn't match ${tensor.descriptor} in $tensor");
        }

        descriptor.setOutputValue(entry.value, value);
      }
    }
  }

  @override
  Differentiator _gradient(Tensor outputTarget, List<String> inputSourceNames,
      Map<String, Tensor> inputs, backPropagatedGradient, String name) {
    throw new UnsupportedError("Gradient of $this");
  }
}

class _AnalyticDifferentiatorImpl extends GroupOperationInternalBase
    implements Differentiator {
  static const String __type = "AnalyticDifferentiator";

  static const String _checkingCountKey = "_CHECKING_COUNT";

  static const String _backPropagatedGradientInputName =
      "backPropagatedGradient";

  final Tensor _target;

  final List<Tensor> _sources;

  num _checkingRate;

  num _checkingDelta;

  num _checkingThreshold;

  _NumericDifferentiatorImpl _numericDifferentiator;

  _AnalyticDifferentiatorImpl(
      this._target, this._sources, backPropagatedGradient,
      {num checkingRate, num checkingDelta, num checkingThreshold, String name})
      : this._checkingRate = checkingRate,
        this._checkingDelta = checkingDelta,
        this._checkingThreshold = checkingThreshold,
        super({_backPropagatedGradientInputName: backPropagatedGradient},
            name ?? _getDifferentialName(_target), __type);

  @override
  Map<Tensor, Tensor> get gradients =>
      new Map.fromIterable(_sources, value: (source) {
        var name = _getDifferentialName(source);
        return hasOutput(name) ? getOutput(name) : null;
      });

  @override
  void buildOperation(GroupDescriptor descriptor) {
    var visitedTensors = _sources
        .expand((source) => _getDifferentiablePathTensors(source, _target))
        .toSet();

    var visitedSources =
        _sources.where((source) => visitedTensors.contains(source)).toList();

    var backPropagatedGradient =
        descriptor.hasInput(_backPropagatedGradientInputName)
            ? descriptor.getInput(_backPropagatedGradientInputName)
            : null;

    for (var source in visitedSources) {
      descriptor.setOutput(
          _getDifferentialName(source),
          _getGradient(
              _target, source, backPropagatedGradient, visitedTensors));
    }

    bool checkGradient = _checkingRate > 0;

    if (checkGradient) {
      _numericDifferentiator = new _NumericDifferentiatorImpl(
          _target, visitedSources,
          delta: _checkingDelta);
    }
  }

  @override
  void computeOperation(OperationDescriptor descriptor) {
    super.computeOperation(descriptor);

    if (!descriptor.isEvaluatingDescriptor) {
      var checkGradient = false;
      if (_checkingRate > 0) {
        int count = _nextCheckingCount;

        if (count * _checkingRate >= 1) {
          checkGradient = true;
          _resetCheckingCount();
        }
      }

      if (checkGradient) {
        var analyticGradients = gradients;
        var numericGradients = _numericDifferentiator.gradients;

        for (var source in _sources) {
          TensorInternalBase analyticGradient = analyticGradients[source];
          TensorInternalBase numericGradient = numericGradients[source];

          var analyticGradientValue = analyticGradient != null
              ? _getInternalOutputValue(analyticGradient.operationOutputName)
              : null;
          var numericGradientValue =
              numericGradient != null ? numericGradient._execute() : null;

          if (analyticGradientValue != null && numericGradientValue != null) {
            var error = (analyticGradientValue - numericGradientValue).abs();

            var errorThreshold =
                (analyticGradientValue * _checkingThreshold).abs();

            if ((error >
                    ((errorThreshold > _checkingThreshold)
                        .select(errorThreshold, _checkingThreshold)))
                .reduceAny()
                .toScalar<bool>()) {
              print("Numeric: $numericGradientValue");
              print("Analytic: $analyticGradientValue");

              throw new StateError(
                  "Bad gradient: $numericGradientValue != $analyticGradientValue in $source [$error]");
            }
          } else if (analyticGradientValue == null &&
              numericGradientValue == null) {
            // skip
          } else {
            throw new StateError(
                "Bad gradient: $numericGradientValue != $analyticGradientValue in $source");
          }
        }
      }
    }
  }

  int get _nextCheckingCount {
    var current = (state.getFromSession(_checkingCountKey) ?? 0) + 1;
    state.setInSession(_checkingCountKey, current);
    return current;
  }

  void _resetCheckingCount() {
    state.removeFromSession(_checkingCountKey);
  }

  Tensor _getGradient(Tensor target, Tensor source,
      Tensor targetBackPropagatedGradient, Set<Tensor> visitedTensors) {
    var gradientName = _getGradientName(target, source);

    if (hasOperation(gradientName)) {
      return getOperation(gradientName).defaultOutput;
    } else {
      var sourceReferences = _getAllReferences(source);

      if (!sourceReferences.contains(target)) {
        var consumers = _getAllConsumers(source);

        var consumerOutputGradients = <Tensor>[];

        consumers.forEach((newSource, consumers) {
          var visitedConsumerOutputs = consumers
              .expand((operation) => operation.outputNames
                  .map((outputName) => operation.getOutput(outputName)))
              .toSet()
              .where((output) => visitedTensors.contains(output));

          for (var consumerOutput in visitedConsumerOutputs) {
            var sourceConsumerInputNames = consumerOutput.operation.inputNames
                .where((consumerInputName) =>
                    consumerOutput.isDifferentiable(consumerInputName))
                .where((consumerInputName) =>
                    consumerOutput.operation.getInput(consumerInputName) ==
                    newSource)
                .toList();

            if (sourceConsumerInputNames.isNotEmpty) {
              var consumerOutputDifferentialName = _getConsumerDifferentialName(
                  target, newSource, consumerOutput);

              var importConsumerOutput = _import(consumerOutput);

              Operation gradient;
              if (hasOperation(consumerOutputDifferentialName)) {
                gradient = getOperation(gradientName);
              } else {
                var backPropagatedGradient = _getGradient(
                    target,
                    consumerOutput,
                    targetBackPropagatedGradient,
                    visitedTensors);

                for (var inputName in consumerOutput.operation.inputNames) {
                  var input = consumerOutput.operation.getInput(inputName);
                  _import(input);
                }

                gradient = importConsumerOutput.operation.gradient(
                    importConsumerOutput.operationOutputName,
                    sourceConsumerInputNames,
                    backPropagatedGradient,
                    name: consumerOutputDifferentialName);
              }

              consumerOutputGradients.addAll(sourceConsumerInputNames.map(
                  (sourceConsumerInputName) => gradient.getOutput(
                      _getDifferentialName(importConsumerOutput.operation
                          .getInput(sourceConsumerInputName)))));
            }
          }
        });

        if (consumerOutputGradients.isNotEmpty) {
          return new Adds(consumerOutputGradients, name: gradientName);
        } else {
          return null;
        }
      } else {
        var importTarget = _import(target);

        return targetBackPropagatedGradient != null
            ? targetBackPropagatedGradient
            : new OnesLike(importTarget,
                dataType: importTarget.dataType, name: gradientName);
      }
    }
  }
}

class _NumericDifferentiatorImpl extends GroupOperationInternalBase
    implements Differentiator {
  static const String __type = "NumericDifferentiator";

  final Tensor _target;

  final List<Tensor> _sources;

  final num _delta;

  _NumericDifferentiatorImpl(this._target, this._sources,
      {num delta, String name})
      : this._delta = delta,
        super(null, name, __type);

  @override
  Map<Tensor, Tensor> get gradients =>
      new Map.fromIterable(_sources, value: (source) {
        var name = _getDifferentialName(source);
        return hasOutput(name) ? getOutput(name) : null;
      });

  @override
  void buildOperation(GroupDescriptor descriptor) {
    var importTarget = _import(_target);

    var importSources = _sources.map((source) => _import(source));

    for (var gradient in _createNumericGradients(importTarget, importSources)) {
      descriptor.setOutput(gradient.operationOutputName, gradient);
    }
  }

  List<Tensor> _createNumericGradients(
          _ImportTensorImpl target, Iterable<_ImportTensorImpl> sources) =>
      sources.map((source) => _createNumericGradient(target, source)).toList();

  Tensor _createNumericGradient(
          _ImportTensorImpl target, _ImportTensorImpl source) =>
      new _NumericGradientImpl(
          target,
          source,
          _delta,
          "${_getDifferentialName(target._importedTensor)}.",
          _getDifferentialName(source._importedTensor));
}

class _NumericGradientImpl extends DefaultTensorBase {
  static const String __type = "NumericGradient";

  static const String _targetInputName = "target";
  static const String _sourceInputName = "source";

  final num _delta;

  _NumericGradientImpl(_ImportTensorImpl target, _ImportTensorImpl source,
      this._delta, String operationName, String outputName)
      : super.output(
            inputs: {_targetInputName: target, _sourceInputName: source},
            operationName: operationName,
            outputName: outputName,
            type: __type);

  @override
  NDObject computeValue(DefaultTensorDescriptor descriptor) {
    var source0 = descriptor.getInputValue(_sourceInputName);

    if (source0 is NDArray) {
      _ImportTensorImpl target = descriptor.getInput(_targetInputName);
      _ImportTensorImpl source = descriptor.getInput(_sourceInputName);

      Tensor newSource = source._importedTensor;

      var sourceRaw =
          source0.reshape(newDimensions: [source0.shape.length]).toVector();

      var gradientRaw = new List(sourceRaw.length);

      for (var i = 0; i < sourceRaw.length; i++) {
        var value = sourceRaw[i];

        sourceRaw[i] = value + _delta / 2;

        var source2 = new NDArray(sourceRaw, dataType: source0.dataType)
            .reshape(newDimensions: source0.shape.dimensions);

        sourceRaw[i] = value - _delta / 2;

        var source1 = new NDArray(sourceRaw, dataType: source0.dataType)
            .reshape(newDimensions: source0.shape.dimensions);

        sourceRaw[i] = value;

        var target2 = run(target, feeds: {newSource: source2});
        var target1 = run(target, feeds: {newSource: source1});

        var dTargetDSource = (target2 - target1) / _delta;

        var sum = dTargetDSource.reduceSum();

        gradientRaw[i] = sum.toScalar();
      }

      return new NDArray(gradientRaw, dataType: source0.dataType)
          .reshape(newDimensions: source0.shape.dimensions);
    } else {
      return source0;
    }
  }
}

class _OperationDescriptorImpl implements OperationDescriptor {
  final OperationInternalBase _operation;

  @override
  final bool isEvaluatingDescriptor;

  _OperationDescriptorImpl(this._operation,
      {this.isEvaluatingDescriptor = false});

  @override
  NDObject toNDObject(value, {@required NDDataType dataType}) {
    var array = toNDArray(value, dataType: dataType);
    return isEvaluatingDescriptor ? array.descriptor : array;
  }

  @override
  Iterable<String> get inputNames => _operation.inputNames;

  @override
  bool hasInput(String name) => _operation.hasInput(name);

  @override
  Tensor getInput(String name) => _operation.getInput(name);

  @override
  NDObject getInputValue(String name) => isEvaluatingDescriptor
      ? _operation._getInputDescriptor(name)
      : _operation._getInputValue(name);

  @override
  set defaultOutputValue(NDObject value) {
    setOutputValue(Operation.defaultOutputName, value);
  }

  @override
  void setOutputValue(String name, NDObject value) {
    if (isEvaluatingDescriptor) {
      _operation._setOutputDescriptor(name, value);
    } else {
      _operation._setOutputValue(name, value);
    }
  }
}

class _GroupTensorsDescriptorImpl implements GroupDescriptor {
  final GroupOperationInternalBase _group;

  final Map<String, TensorInternalBase> _internalInputs;

  final Map<String, dynamic> _internalOutputs = {};

  final List<ExecutableBase> _internalExecutables = [];

  _GroupTensorsDescriptorImpl(this._group, this._internalInputs);

  @override
  Iterable<String> get inputNames => _internalInputs.keys;

  @override
  bool hasInput(String name) => _internalInputs.containsKey(name);

  @override
  Tensor getInput(String name) =>
      _internalInputs[name] ??
      (throw new ArgumentError.value(
          name, "Input not specified in $_group descriptor"));

  @override
  bool hasImport(Executable executable) => _group._hasImport(executable);

  @override
  E import<E extends Executable>(E executable) => _group._import(executable);

  @override
  set defaultOutput(Tensor defaultOutput) {
    setOutput(Operation.defaultOutputName, defaultOutput);
  }

  @override
  void setOutput(String name, Tensor output) {
    if (output != null) {
      _internalOutputs[name] = output;
    } else {
      throw new ArgumentError.notNull("Output $name in $_group descriptor");
    }
  }

  @override
  void addExecutable(Executable executable) {
    if (executable != null) {
      _internalExecutables.add(executable);
    } else {
      throw new ArgumentError.notNull("Executable in $_group descriptor");
    }
  }
}

class _GradientsComputersDescriptorImpl
    implements GradientsComputersDescriptor {
  final OperationInternalBase _operation;

  final Map<String, Map<String, TensorGradientComputer>> _gradientsComputers =
      {};

  _GradientsComputersDescriptorImpl(this._operation);

  @override
  Iterable<String> get inputNames => _operation.inputNames;

  @override
  bool hasInput(String name) => _operation.hasInput(name);

  @override
  Tensor getInput(String name) => _operation.getInput(name);

  @override
  void setDefaultOutputGradient(
      String inputName, TensorGradientComputer gradientComputer) {
    setOutputGradient(Operation.defaultOutputName, inputName, gradientComputer);
  }

  @override
  void setOutputGradient(String outputName, String inputName,
      TensorGradientComputer gradientComputer) {
    _gradientsComputers.putIfAbsent(outputName, () => {})[inputName] =
        gradientComputer;
  }

  bool _hasGradientComputer(String outputName, String inputName) =>
      _gradientsComputers.containsKey(outputName) &&
      _gradientsComputers[outputName].containsKey(inputName);

  Map<String, TensorGradientComputer> _getComputerGradients(String outputName) {
    return _gradientsComputers[outputName] ?? {};
  }
}

class _TensorGradientDescriptorImpl implements TensorGradientDescriptor {
  final _GradientOperationImpl _operation;

  final OperationDescriptor _operationDescriptor;

  _TensorGradientDescriptorImpl(this._operation, this._operationDescriptor);

  @override
  bool get isEvaluatingDescriptor =>
      _operationDescriptor.isEvaluatingDescriptor;

  @override
  NDObject toNDObject(value, {NDDataType dataType}) =>
      _operationDescriptor.toNDObject(value, dataType: dataType);

  @override
  Iterable<String> get inputNames =>
      _operationDescriptor.inputNames.where((inputName) =>
          inputName != _GradientOperationImpl._outputInputName &&
          inputName != _GradientOperationImpl._backPropagatedGradientInputName);

  @override
  bool hasInput(String name) =>
      name != _GradientOperationImpl._outputInputName &&
      name != _GradientOperationImpl._backPropagatedGradientInputName &&
      _operationDescriptor.hasInput(name);

  @override
  Tensor getInput(String name) => hasInput(name)
      ? _operationDescriptor.getInput(name)
      : (throw new ArgumentError.value(
          name, "Input not specified in $_operation descriptor"));

  @override
  NDObject getInputValue(String name) => hasInput(name)
      ? _operationDescriptor.getInputValue(name)
      : (throw new ArgumentError.value(
          name, "Input not specified in $_operation descriptor"));

  @override
  Tensor get output =>
      _operationDescriptor.getInput(_GradientOperationImpl._outputInputName);

  @override
  NDObject get outputValue => _operationDescriptor
      .getInputValue(_GradientOperationImpl._outputInputName);

  @override
  Tensor get backPropagatedGradient => _operationDescriptor
      .getInput(_GradientOperationImpl._backPropagatedGradientInputName);

  @override
  NDObject get backPropagatedGradientValue => !output.isFeedValue
      ? _operationDescriptor.getInputValue(
          _GradientOperationImpl._backPropagatedGradientInputName)
      : new NDArray.zeros(outputValue.shape.dimensions,
          dataType: _operationDescriptor
              .getInputValue(
                  _GradientOperationImpl._backPropagatedGradientInputName)
              .dataType);
}

String _getConsumerDifferentialName(
        Tensor target, Tensor source, Tensor consumerOutput) =>
    "${_getGradientName(target, source)}.${_getFlatPath(consumerOutput)}";

String _getGradientName(Tensor target, Tensor source) =>
    "${_getDifferentialName(target)}.${_getDifferentialName(source)}";

String _getDifferentialName(Tensor tensor) => "d${_getFlatPath(tensor)}";

String _getFlatPath(Executable executable) {
  if (executable is Tensor) {
    return "${_getFlatPath(executable.operation)}${!executable.isDefaultOutput ? ".${executable.operationOutputName}" : ""}";
  } else {
    return executable.path.replaceAll("/", ".");
  }
}

Set<Tensor> _getDifferentiablePathTensors(Tensor from, Tensor to) {
  var newPath = new Set<Tensor>();

  if (from == to) {
    newPath.add(from);
  } else {
    List<Tensor> nextTensors = [];

    if (to.operation is GroupOperationInternalBase) {
      GroupOperationInternalBase group = to.operation;

      nextTensors.add(group._getInternalOutput(to.operationOutputName));
    } else if (to is _GroupInternalInputTensorImpl) {
      nextTensors.add(to._externalInput);
    } else {
      for (var inputName in to.operation.inputNames.where((inputName) =>
          to.operation.isDifferentiable(to.operationOutputName, inputName))) {
        var input = to.operation.getInput(inputName);

        nextTensors.add(input);
      }
    }

    for (var tensor in nextTensors) {
      var path = _getDifferentiablePathTensors(from, tensor);
      if (path.isNotEmpty) {
        if (newPath.isEmpty) {
          newPath.add(to);
        }
        newPath.addAll(path);
      }
    }
  }

  return newPath;
}

Set<Tensor> _getAllReferences(Tensor source) {
  var references = new Set<Tensor>();

  references.add(source);

  if (source.graph is GroupOperationInternalBase) {
    GroupOperationInternalBase sourceGraph = source.graph;

    references.addAll(sourceGraph._internalOutputs.keys.expand((outputName) {
      var internalOutput = sourceGraph._getInternalOutput(outputName);

      if (internalOutput == source) {
        return _getAllReferences(sourceGraph.getOutput(outputName));
      } else {
        return [];
      }
    }));
  }

  if (source.operation is GroupOperationInternalBase) {
    GroupOperationInternalBase group = source.operation;

    var internalInputs = group.inputNames
        .where((inputName) => group.getInput(inputName) == source)
        .map((inputName) => group._getInternalInput(inputName));

    for (var internalSource in internalInputs) {
      references.addAll(_getAllReferences(internalSource));
    }
  }

  return references;
}

Map<Tensor, Set<Operation>> _getAllConsumers(Tensor startSource) {
  var sources = [startSource];

  if (startSource.graph is GroupOperationInternalBase) {
    GroupOperationInternalBase sourceGraph = startSource.graph;

    sources.addAll(sourceGraph._internalOutputs.keys.expand((outputName) {
      var internalOutput = sourceGraph._getInternalOutput(outputName);

      if (internalOutput == startSource) {
        return [sourceGraph.getOutput(outputName)];
      } else {
        return [];
      }
    }));
  }

  var allConsumers = <Tensor, Set<Operation>>{};

  for (var source in sources) {
    GraphBase sourceGraph = source.graph;

    var consumers = source.consumerIds
        .map((operationId) => sourceGraph._getOperation(operationId));

    for (var consumer in consumers) {
      if (consumer is GroupOperationInternalBase) {
        var internalInputs = consumer.inputNames
            .where((inputName) => consumer.getInput(inputName) == source)
            .map((inputName) => consumer._getInternalInput(inputName));

        for (var internalSource in internalInputs) {
          allConsumers.addAll(_getAllConsumers(internalSource));
        }
      } else {
        var sourceConsumers = allConsumers.putIfAbsent(source, () => new Set());
        sourceConsumers.add(consumer);
      }
    }
  }

  return allConsumers;
}

class _State {
  final Map<String, dynamic> _values = {};

  bool contains(String key) => _values.containsKey(key);

  void remove(String key) {
    _values.remove(key);
  }

  dynamic operator [](String key) => _values[key];

  void operator []=(String key, value) {
    _values[key] = value;
  }
}

class _SessionState extends _State {
  final Map<Executable, _State> _states = {};

  bool __isClosed = false;

  void _close() {
    __isClosed = true;
  }

  bool get _isClosed => __isClosed;

  _State _getSessionState(Executable executable) =>
      _states.putIfAbsent(executable, () => new _State());
}

class _ExecutionState extends _State {
  final _SessionState _sessionState;

  final Map<Tensor, NDArray> _feeds;

  final Map<Executable, ExecutableState> _states = {};

  final _ExecutionState _previous;

  _ExecutionState(
      this._sessionState, Map<Tensor, dynamic> feeds, this._previous)
      : this._feeds = mapMap<Tensor, dynamic, Tensor, NDArray>(feeds,
            value: (key, value) {
          var arrayValue = toNDArray(value, dataType: key.dataType);

          if (arrayValue.descriptor.isCompatibleWith(key.descriptor)) {
            return arrayValue;
          } else {
            throw new ArgumentError(
                "Feed descriptor ${arrayValue.descriptor} doesn't match ${key.descriptor}");
          }
        });

  _ExecutableStateImpl _getState(Executable executable) {
    dynamic target = executable;

    if (target is _ImportOperationImpl) {
      target = target._importedOperation;
    } else if (target is _ImportTensorImpl) {
      target = target._importedTensor;
    }

    return _states.putIfAbsent(target, () {
      if (target is Tensor) {
        return new _TensorStateImpl(
            _sessionState._getSessionState(target), _getFeed(target));
      } else {
        return new _OperationStateImpl(_sessionState._getSessionState(target));
      }
    });
  }

  NDArray _getFeed(Tensor tensor) {
    dynamic target = tensor;

    if (target is _ImportTensorImpl) {
      target = target._importedTensor;
    }

    return _feeds[target] ?? _previous?._getFeed(target);
  }
}

abstract class _ExecutableStateImpl extends _State implements ExecutableState {
  final _State _sessionExecutableState;

  _ExecutableStateImpl(this._sessionExecutableState);

  @override
  bool get isNotExecuted => !isExecuted;

  @override
  bool containsInSession(String key) => _sessionExecutableState.contains(key);

  @override
  dynamic getFromSession(String key) => _sessionExecutableState[key];

  @override
  void setInSession(String key, value) {
    _sessionExecutableState[key] = value;
  }

  @override
  void removeFromSession(String key) {
    _sessionExecutableState.remove(key);
  }
}

class _OperationStateImpl extends _ExecutableStateImpl
    implements OperationState {
  static const String _executedStateKey = "_EXECUTED";

  _OperationStateImpl(_State sessionExecutableState)
      : super(sessionExecutableState);

  @override
  bool get isExecuted => this.contains(_executedStateKey);

  void setExecuted() {
    this[_executedStateKey] = true;
  }
}

class _TensorStateImpl extends _ExecutableStateImpl implements TensorState {
  static const String _valueStateKey = "_VALUE";

  final NDArray _feed;

  _TensorStateImpl(_State sessionExecutableState, this._feed)
      : super(sessionExecutableState);

  @override
  bool get isExecuted => isExecutionValue;

  @override
  bool get isEvaluated => value != null;

  @override
  bool get isNotEvaluated => value == null;

  @override
  bool get isExecutionValue => !isFeedValue && this.contains(_valueStateKey);

  @override
  bool get isFeedValue => _feed != null;

  NDArray get value => _feed ?? this[_valueStateKey];

  set value(NDArray value) {
    this[_valueStateKey] = value;
  }
}
