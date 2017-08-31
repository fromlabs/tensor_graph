void main() {
  assert(debug("ciao"));
}

bool debug(Object object) {
  print(object);
  return true;
}