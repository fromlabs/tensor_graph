// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

class EnumerationEntry<T> {
  final int index;
  final T element;

  EnumerationEntry(this.index, this.element);
}

class MapEntry<K, V> {
  final K key;
  final V value;

  MapEntry(this.key, this.value);
}

Iterable<EnumerationEntry<T>> enumerate<T>(Iterable<T> iterable) {
  var index = 0;
  return iterable.map((element) => new EnumerationEntry(index++, element));
}

Iterable<int> range(int from, int to) =>
    new Iterable<int>.generate(to - from + 1, (i) => from + i);

Iterable<MapEntry<K, V>> entries<K, V>(Map<K, V> map) => map.keys
    .map((K key) => new MapEntry(key, map[key]));
