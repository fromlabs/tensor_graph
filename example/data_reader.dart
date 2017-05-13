import "dart:async";
import "dart:collection";
import 'dart:math';

Future main() async {
  StreamController<List<int>> streamController;

  try {
    streamController = new StreamController<List<int>>();

    streamController.add([1, 2, 3]);
    streamController.add([4, 5, 6]);

    var reader = new DataStreamReader(streamController.stream);

    log(await reader.nextData(2));

    log(await reader.nextData(2));

    log(await reader.nextData(2));
  } finally {
    await streamController.close();
  }
}

void log(List<int> data) {
  for (var i = 0; i < data.length; i++) {
    print("$i: ${data[i]}");
  }
}

class DataList implements List<int> {
  final List<List<int>> _chunks;
  final int _firstChunkOffset;
  final int _length;

  DataList(List<int> chunk, this._firstChunkOffset, this._length)
      : this._chunks = [chunk];

  DataList.multi(this._chunks, this._firstChunkOffset, this._length);

  @override
  int get length => _length;

  @override
  set length(int length) {
    // TODO to implement DataList.length
    throw new UnimplementedError("to implement DataList.length: $this");
  }

  @override
  int operator [](int index) {
    var chunkIndex = _firstChunkOffset + index;

    for (var chunk in _chunks) {
      if (chunkIndex < chunk.length) {
        return chunk[chunkIndex];
      } else {
        chunkIndex -= chunk.length;
      }
    }

    throw new ArgumentError();
  }

  @override
  void operator []=(int index, int value) {
    // TODO to implement DataList.[]=
    throw new UnimplementedError("to implement DataList.[]=: $this");
  }

  @override
  void add(int value) {
    // TODO to implement DataList.add
    throw new UnimplementedError("to implement DataList.add: $this");
  }

  @override
  void addAll(Iterable<int> iterable) {
    // TODO to implement DataList.addAll
    throw new UnimplementedError("to implement DataList.addAll: $this");
  }

  @override
  bool any(bool f(int element)) {
    // TODO to implement DataList.any
    throw new UnimplementedError("to implement DataList.any: $this");
  }

  @override
  Map<int, int> asMap() {
    // TODO to implement DataList.asMap
    throw new UnimplementedError("to implement DataList.asMap: $this");
  }

  @override
  void clear() {
    // TODO to implement DataList.clear
    throw new UnimplementedError("to implement DataList.clear: $this");
  }

  @override
  bool contains(Object element) {
    // TODO to implement DataList.contains
    throw new UnimplementedError("to implement DataList.contains: $this");
  }

  @override
  int elementAt(int index) {
    // TODO to implement DataList.elementAt
    throw new UnimplementedError("to implement DataList.elementAt: $this");
  }

  @override
  bool every(bool f(int element)) {
    // TODO to implement DataList.every
    throw new UnimplementedError("to implement DataList.every: $this");
  }

  @override
  Iterable<T> expand<T>(Iterable<T> f(int element)) {
    // TODO to implement DataList.expand
    throw new UnimplementedError("to implement DataList.expand: $this");
  }

  @override
  void fillRange(int start, int end, [int fillValue]) {
    // TODO to implement DataList.fillRange
    throw new UnimplementedError("to implement DataList.fillRange: $this");
  }

  // TODO: implement first
  @override
  int get first {
    // TODO to implement DataList.first
    throw new UnimplementedError("to implement DataList.first: $this");
  }

  @override
  int firstWhere(bool test(int element), {int orElse()}) {
    // TODO to implement DataList.firstWhere
    throw new UnimplementedError("to implement DataList.firstWhere: $this");
  }

  @override
  T fold<T>(T initialValue, T combine(T previousValue, int element)) {
    // TODO to implement DataList.fold
    throw new UnimplementedError("to implement DataList.fold: $this");
  }

  @override
  void forEach(void f(int element)) {
    // TODO to implement DataList.forEach
    throw new UnimplementedError("to implement DataList.forEach: $this");
  }

  @override
  Iterable<int> getRange(int start, int end) {
    // TODO to implement DataList.getRange
    throw new UnimplementedError("to implement DataList.getRange: $this");
  }

  @override
  int indexOf(int element, [int start = 0]) {
    // TODO to implement DataList.indexOf
    throw new UnimplementedError("to implement DataList.indexOf: $this");
  }

  @override
  void insert(int index, int element) {
    // TODO to implement DataList.insert
    throw new UnimplementedError("to implement DataList.insert: $this");
  }

  @override
  void insertAll(int index, Iterable<int> iterable) {
    // TODO to implement DataList.insertAll
    throw new UnimplementedError("to implement DataList.insertAll: $this");
  }

  // TODO: implement isEmpty
  @override
  bool get isEmpty {
    // TODO to implement DataList.isEmpty
    throw new UnimplementedError("to implement DataList.isEmpty: $this");
  }

  // TODO: implement isNotEmpty
  @override
  bool get isNotEmpty {
    // TODO to implement DataList.isNotEmpty
    throw new UnimplementedError("to implement DataList.isNotEmpty: $this");
  }

  // TODO: implement iterator
  @override
  Iterator<int> get iterator {
    // TODO to implement DataList.iterator
    throw new UnimplementedError("to implement DataList.iterator: $this");
  }

  @override
  String join([String separator = ""]) {
    // TODO to implement DataList.join
    throw new UnimplementedError("to implement DataList.join: $this");
  }

  // TODO: implement last
  @override
  int get last {
    // TODO to implement DataList.last
    throw new UnimplementedError("to implement DataList.last: $this");
  }

  @override
  int lastIndexOf(int element, [int start]) {
    // TODO to implement DataList.lastIndexOf
    throw new UnimplementedError("to implement DataList.lastIndexOf: $this");
  }

  @override
  int lastWhere(bool test(int element), {int orElse()}) {
    // TODO to implement DataList.lastWhere
    throw new UnimplementedError("to implement DataList.lastWhere: $this");
  }

  @override
  Iterable<T> map<T>(T f(int e)) {
    // TODO to implement DataList.map
    throw new UnimplementedError("to implement DataList.map: $this");
  }

  @override
  int reduce(int combine(int value, int element)) {
    // TODO to implement DataList.reduce
    throw new UnimplementedError("to implement DataList.reduce: $this");
  }

  @override
  bool remove(Object value) {
    // TODO to implement DataList.remove
    throw new UnimplementedError("to implement DataList.remove: $this");
  }

  @override
  int removeAt(int index) {
    // TODO to implement DataList.removeAt
    throw new UnimplementedError("to implement DataList.removeAt: $this");
  }

  @override
  int removeLast() {
    // TODO to implement DataList.removeLast
    throw new UnimplementedError("to implement DataList.removeLast: $this");
  }

  @override
  void removeRange(int start, int end) {
    // TODO to implement DataList.removeRange
    throw new UnimplementedError("to implement DataList.removeRange: $this");
  }

  @override
  void removeWhere(bool test(int element)) {
    // TODO to implement DataList.removeWhere
    throw new UnimplementedError("to implement DataList.removeWhere: $this");
  }

  @override
  void replaceRange(int start, int end, Iterable<int> replacement) {
    // TODO to implement DataList.replaceRange
    throw new UnimplementedError("to implement DataList.replaceRange: $this");
  }

  @override
  void retainWhere(bool test(int element)) {
    // TODO to implement DataList.retainWhere
    throw new UnimplementedError("to implement DataList.retainWhere: $this");
  }

  // TODO: implement reversed
  @override
  Iterable<int> get reversed {
    // TODO to implement DataList.reversed
    throw new UnimplementedError("to implement DataList.reversed: $this");
  }

  @override
  void setAll(int index, Iterable<int> iterable) {
    // TODO to implement DataList.setAll
    throw new UnimplementedError("to implement DataList.setAll: $this");
  }

  @override
  void setRange(int start, int end, Iterable<int> iterable,
      [int skipCount = 0]) {
    // TODO to implement DataList.setRange
    throw new UnimplementedError("to implement DataList.setRange: $this");
  }

  @override
  void shuffle([Random random]) {
    // TODO to implement DataList.shuffle
    throw new UnimplementedError("to implement DataList.shuffle: $this");
  }

  // TODO: implement single
  @override
  int get single {
    // TODO to implement DataList.single
    throw new UnimplementedError("to implement DataList.single: $this");
  }

  @override
  int singleWhere(bool test(int element)) {
    // TODO to implement DataList.singleWhere
    throw new UnimplementedError("to implement DataList.singleWhere: $this");
  }

  @override
  Iterable<int> skip(int count) {
    // TODO to implement DataList.skip
    throw new UnimplementedError("to implement DataList.skip: $this");
  }

  @override
  Iterable<int> skipWhile(bool test(int value)) {
    // TODO to implement DataList.skipWhile
    throw new UnimplementedError("to implement DataList.skipWhile: $this");
  }

  @override
  void sort([int compare(int a, int b)]) {
    // TODO to implement DataList.sort
    throw new UnimplementedError("to implement DataList.sort: $this");
  }

  @override
  List<int> sublist(int start, [int end]) {
    // TODO to implement DataList.sublist
    throw new UnimplementedError("to implement DataList.sublist: $this");
  }

  @override
  Iterable<int> take(int count) {
    // TODO to implement DataList.take
    throw new UnimplementedError("to implement DataList.take: $this");
  }

  @override
  Iterable<int> takeWhile(bool test(int value)) {
    // TODO to implement DataList.takeWhile
    throw new UnimplementedError("to implement DataList.takeWhile: $this");
  }

  @override
  List<int> toList({bool growable: true}) {
    // TODO to implement DataList.toList
    throw new UnimplementedError("to implement DataList.toList: $this");
  }

  @override
  Set<int> toSet() {
    // TODO to implement DataList.toSet
    throw new UnimplementedError("to implement DataList.toSet: $this");
  }

  @override
  Iterable<int> where(bool test(int element)) {
    // TODO to implement DataList.where
    throw new UnimplementedError("to implement DataList.where: $this");
  }
}

class DataStreamReader {
  final Stream<List<int>> _stream;

  Completer _waitingChunkCompleter;

  final Queue<List<int>> _chunkQueue = new Queue<List<int>>();
  int _currentChunkOffset = 0;

  DataStreamReader(this._stream) {
    this._stream.listen(_onData);
  }

  Future<int> nextSingleData() {
    var dataOrFuture = nextSingleDataOrFuture();

    if (dataOrFuture is Future<int>) {
      return dataOrFuture;
    } else {
      return new Future.value(dataOrFuture);
    }
  }

  Future<List<int>> nextData(int length) {
    var dataOrFuture = nextDataOrFuture(length);

    if (dataOrFuture is Future<List<int>>) {
      return dataOrFuture;
    } else {
      return new Future.value(dataOrFuture);
    }
  }

  Future<int> get nextInt32 {
    var dataOrFuture = nextDataOrFuture(4);

    if (dataOrFuture is Future<List<int>>) {
      return dataOrFuture.then((data) => _toInt32(data));
    } else {
      return new Future.value(_toInt32(dataOrFuture));
    }
  }

  FutureOr<int> nextSingleDataOrFuture() {
    var dataOrFuture = nextDataOrFuture(1);

    if (dataOrFuture is Future<List<int>>) {
      return dataOrFuture.then((data) => data[0]);
    } else {
      List<int> data = dataOrFuture;
      return data[0];
    }
  }

  FutureOr<List<int>> nextDataOrFuture(int length) {
    var chunkOrFuture = _currentChunkOrFuture();

    var chunks = [];
    var left = length;
    var firstChunkOffset;
    if (chunkOrFuture is! Future<List<int>>) {
      List<int> chunk = chunkOrFuture;
      firstChunkOffset = _currentChunkOffset;
      left -= _consumeCurrentChunk(left);
      if (left == 0) {
        return new DataList(chunk, firstChunkOffset, length);
      } else {
        chunks.add(chunk);
      }
    }

    return new Future(() {
      if (chunkOrFuture is Future<List<int>>) {
        return chunkOrFuture.then((chunk) {
          firstChunkOffset = _currentChunkOffset;
          left -= _consumeCurrentChunk(left);
          chunks.add(chunk);
        });
      }
    }).then((_) async {
      while (left > 0) {
        var chunk = await _currentChunk();
        left -= _consumeCurrentChunk(left);
        chunks.add(chunk);
      }

      return new DataList.multi(chunks, firstChunkOffset, length);
    });
  }

  Future<List<int>> _currentChunk() {
    var chunkOrFuture = _currentChunkOrFuture();

    if (chunkOrFuture is Future<List<int>>) {
      return chunkOrFuture;
    } else {
      return new Future.value(chunkOrFuture);
    }
  }

  FutureOr<List<int>> _currentChunkOrFuture() {
    if (_chunkQueue.isEmpty) {
      _waitingChunkCompleter = new Completer();
      return _waitingChunkCompleter.future.then((_) => _chunkQueue.first);
    } else {
      return _chunkQueue.first;
    }
  }

  void _onData(List<int> chunk) {
    _chunkQueue.add(chunk);

    if (_waitingChunkCompleter != null) {
      var completer = _waitingChunkCompleter;
      _waitingChunkCompleter = null;
      completer.complete();
    }
  }

  int _consumeCurrentChunk(int length) {
    var chunk = _chunkQueue.first;
    var left = chunk.length - _currentChunkOffset;

    if (length <= left) {
      _currentChunkOffset += length;
      return length;
    } else {
      _currentChunkOffset = 0;
      _chunkQueue.removeFirst();
      return left;
    }
  }

  int _toInt32(List<int> data32) =>
      (data32[0] << 24) + (data32[1] << 16) + (data32[2] << 8) + data32[3];
}
