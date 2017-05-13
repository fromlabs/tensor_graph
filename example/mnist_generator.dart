import "dart:async";
import "dart:io";

import "package:http/http.dart" as http;
import "package:path/path.dart" as path;

import "data_reader.dart";

Future main() async {
  await createDataset();
}

Future<Map<String, Map<String, List>>> createDataset() async {
  await downloadData("dataset/mnist",
      "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz");
  await downloadData("dataset/mnist",
      "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz");
  await downloadData("dataset/mnist",
      "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz");
  await downloadData("dataset/mnist",
      "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz");

  var trainLabels = await readLabels("dataset/mnist/train-labels-idx1-ubyte");
  var testLabels = await readLabels("dataset/mnist/t10k-labels-idx1-ubyte");
  var trainImages = await readImages("dataset/mnist/train-images-idx3-ubyte");
  var testImages = await readImages("dataset/mnist/t10k-images-idx3-ubyte");

  return {
    "train": {"images": trainImages, "labels": trainLabels},
    "test": {"images": testImages, "labels": testLabels}
  };
}

Future downloadData(String targetDir, String url) async {
  var gzName = path.split(url).last;
  var unpackedName = gzName.substring(0, gzName.lastIndexOf("."));

  var unpackedFile = new File("$targetDir/$unpackedName");

  if (!unpackedFile.existsSync()) {
    print("Download $url in $targetDir");

    var response = await http.get(url);

    unpackedFile.createSync(recursive: true);

    unpackedFile.writeAsBytesSync(GZIP.decode(response.bodyBytes));
  }
}

Future<List<List<int>>> readLabels(String path) async {
  var file = new File(path);

  var stream = file.openRead();

  var reader = new DataStreamReader(stream);

  var data = await reader.nextData(4);

  if (data[0] != 0x00 ||
      data[1] != 0x00 ||
      data[2] != 0x08 ||
      data[3] != 0x01) {
    throw new ArgumentError("Wrong magic number");
  }

  var count = await reader.nextInt32;

  if (count != file.lengthSync() - 8) {
    throw new ArgumentError("Invalid length");
  }

  List<List<int>> labels = new List(count);

  for (var i = 0; i < count; i++) {
    var dataOrFuture = reader.nextSingleDataOrFuture();

    var data;
    if (dataOrFuture is Future<int>) {
      data = await dataOrFuture;
    } else {
      data = dataOrFuture;
    }

    List<int> row = new List.filled(10, 0);
    row[data] = 1;

    labels[i] = row;
  }

  return labels;
}

Future<List<List<int>>> readImages(String path) async {
  var file = new File(path);

  var stream = file.openRead();

  var reader = new DataStreamReader(stream);

  var data = await reader.nextData(4);

  if (data[0] != 0x00 ||
      data[1] != 0x00 ||
      data[2] != 0x08 ||
      data[3] != 0x03) {
    throw new ArgumentError("Wrong magic number");
  }

  var count = await reader.nextInt32;
  var rows = await reader.nextInt32;
  var columns = await reader.nextInt32;

  if (count * rows * columns != file.lengthSync() - 16) {
    throw new ArgumentError("Invalid length");
  }

  if (rows != 28) {
    throw new ArgumentError("Invalid rows");
  }

  if (columns != 28) {
    throw new ArgumentError("Invalid columns");
  }

  List<List<int>> images = new List(count);

  int length = rows * columns;

  for (var i = 0; i < count; i++) {
    var dataOrFuture = reader.nextDataOrFuture(length);

    var data;
    if (dataOrFuture is Future<List<int>>) {
      data = await dataOrFuture;
    } else {
      data = dataOrFuture;
    }
    images[i] = data;
  }

  return images;
}
