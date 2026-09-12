function processAll(items, callback) {
  return items.map(function (item) {
    return transform(item, callback);
  });
}

function transform(item, callback) {
  return callback(item);
}
