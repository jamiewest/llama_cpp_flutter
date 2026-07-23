/// [bytes] as a short human-readable size, e.g. `1.2 GB`.
String formatBytes(int bytes) {
  if (bytes <= 0) return '0 MB';
  const units = <String>['B', 'KB', 'MB', 'GB', 'TB'];
  var value = bytes.toDouble();
  var unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return '${value.toStringAsFixed(value < 10 && unit > 1 ? 1 : 0)} '
      '${units[unit]}';
}
