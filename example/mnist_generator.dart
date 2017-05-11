import "dart:async";
import "dart:io";
import "dart:convert";

Future main() async {
  var trainLabels = await readLabels("dataset/mnist/train-labels-idx1-ubyte");
  var testLabels = await readLabels("dataset/mnist/t10k-labels-idx1-ubyte");
  var trainImages = await readImages("dataset/mnist/train-images-idx3-ubyte");
  var testImages = await readImages("dataset/mnist/t10k-images-idx3-ubyte");

  var json = new File("dataset/mnist/dataset.json");
  json.writeAsStringSync(JSON.encode({
    "train": {"images": trainImages, "labels": trainLabels},
    "test": {"images": testImages, "labels": testLabels}
  }));
}

Future<List<List<int>>> readLabels(String path) async {
  var file = new File(path);

  var result = [];
  await for (var data in file.openRead()) {
    for (int i = 0; i < data.length; i++) {
      result.add(data[i]);
    }
  }

  if (result[0] != 0x00 ||
      result[1] != 0x00 ||
      result[2] != 0x08 ||
      result[3] != 0x01) {
    throw new ArgumentError("Wrong magic number");
  }

  var count =
      (result[4] << 24) + (result[5] << 16) + (result[6] << 8) + result[7];

  if (count != file.lengthSync() - 8) {
    throw new ArgumentError("Invalid length");
  }

  var labels = [];

  var i = 0;
  var offset = 8;
  while (i < count) {
    List<int> row = new List.filled(10, 0);
    row[result[offset]] = 1;

    labels.add(row);

    offset++;
    i++;
  }

  return labels;
}

Future<List<List<int>>> readImages(String path) async {
  var file = new File(path);

  var result = [];
  await for (var data in file.openRead()) {
    for (int i = 0; i < data.length; i++) {
      result.add(data[i]);
    }
  }

  if (result[0] != 0x00 ||
      result[1] != 0x00 ||
      result[2] != 0x08 ||
      result[3] != 0x03) {
    throw new ArgumentError("Wrong magic number");
  }

  var count =
      (result[4] << 24) + (result[5] << 16) + (result[6] << 8) + result[7];

  var rows =
      (result[8] << 24) + (result[9] << 16) + (result[10] << 8) + result[11];

  var columns =
      (result[12] << 24) + (result[13] << 16) + (result[14] << 8) + result[15];

  if (count * rows * columns != file.lengthSync() - 16) {
    throw new ArgumentError("Invalid length");
  }

  if (rows != 28) {
    throw new ArgumentError("Invalid rows");
  }

  if (columns != 28) {
    throw new ArgumentError("Invalid columns");
  }

  List<List<int>> images = [];

  var i = 0;
  var offset = 16;
  while (i < count) {
    List<int> image = [];
    for (var i2 = 0; i2 < rows * columns; i2++) {
      image.add(result[offset]);

      offset++;
    }

    images.add(image);
    i++;
  }

  return images;
}
