// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

import "dart:math";

List<E> extractBatchFromIndexes<E>(List<E> data, List<int> indexes) =>
    indexes.map<E>((index) => data[index]).toList();

class BatchGenerator {
  final int _count;
  final Random _random;

  int _offset;
  List<int> _data;

  BatchGenerator(this._count, [Random random])
      : _random = random ?? new Random();

  List<int> getBatchIndexes(int batchSize) {
    if (_data == null) {
      _offset = 0;
      _data = new List.generate(_count, (index) => index);
      _data.shuffle(_random);
    }

    var batch;
    var end = _offset + batchSize;
    if (end < _data.length) {
      batch = _data.sublist(_offset, end);
      _offset = end;
    } else {
      batch = [];
      batch.addAll(_data.sublist(_offset));
      var leftCount = end - _data.length;
      _data = null;
      batch.addAll(getBatchIndexes(leftCount));
    }
    return batch;
  }
}
