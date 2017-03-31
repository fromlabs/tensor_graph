// Copyright (c) 2017 Roberto Tassi. All rights reserved. Use of this source code
// is governed by a MIT-style license that can be found in the LICENSE file.

abstract class Graph {
  Iterable<String> get exportedIds;

  Iterable<String> get importingIds;
}
