import 'dart:convert';
import 'dart:typed_data';

import 'contract_failure.dart';

const int maxRealtimeCoachingEventBytes = 65536;

/// Contract JSON is shallow (currently under ten levels). A limit of 64 keeps
/// malformed recursive input deterministic on every supported Dart runtime.
const int maxCanonicalJsonDepth = 64;

Object? parseCanonicalJson(Uint8List bytes) {
  if (bytes.length > maxRealtimeCoachingEventBytes) {
    throw const ContractFailure(
      ContractFailureCode.sizeLimitExceeded,
      'realtime coaching event exceeds maximum wire size',
    );
  }

  late final String source;
  try {
    source = utf8.decode(bytes, allowMalformed: false);
  } on FormatException {
    throw const ContractFailure(
      ContractFailureCode.invalidUtf8,
      'realtime coaching event is not valid UTF-8',
    );
  }

  final parser = _StrictJsonParser(source);
  final value = parser.parseDocument();
  final canonical = canonicalJsonBytes(value);
  if (!_bytesEqual(canonical, bytes)) {
    throw const ContractFailure(
      ContractFailureCode.nonCanonicalJson,
      'realtime coaching event JSON must be canonical',
    );
  }
  return value;
}

Uint8List canonicalJsonBytes(Object? value) =>
    Uint8List.fromList(utf8.encode(_canonicalJson(value, 0)));

String _canonicalJson(Object? value, int depth) {
  _requireSafeDepth(depth);
  if (value == null) return 'null';
  if (value is bool) return value ? 'true' : 'false';
  if (value is int) return value.toString();
  if (value is BigInt) return value.toString();
  if (value is String) return jsonEncode(value);
  if (value is List<Object?>) {
    return '[${value.map((item) => _canonicalJson(item, depth + 1)).join(',')}]';
  }
  if (value is Map<String, Object?>) {
    final keys = value.keys.toList()..sort(_compareUnicodeCodePoints);
    return '{${keys.map((key) => '${jsonEncode(key)}:${_canonicalJson(value[key], depth + 1)}').join(',')}}';
  }
  throw const ContractFailure(
    ContractFailureCode.invalidValue,
    'unsupported canonical JSON value',
  );
}

void _requireSafeDepth(int depth) {
  if (depth > maxCanonicalJsonDepth) {
    throw const ContractFailure(
      ContractFailureCode.invalidJson,
      'JSON nesting exceeds the contract safety limit',
    );
  }
}

int _compareUnicodeCodePoints(String first, String second) {
  final firstRunes = first.runes.iterator;
  final secondRunes = second.runes.iterator;
  while (true) {
    final hasFirst = firstRunes.moveNext();
    final hasSecond = secondRunes.moveNext();
    if (!hasFirst || !hasSecond) {
      if (hasFirst == hasSecond) return 0;
      return hasFirst ? 1 : -1;
    }
    final comparison = firstRunes.current.compareTo(secondRunes.current);
    if (comparison != 0) return comparison;
  }
}

bool _bytesEqual(Uint8List first, Uint8List second) {
  if (first.length != second.length) return false;
  for (var index = 0; index < first.length; index += 1) {
    if (first[index] != second[index]) return false;
  }
  return true;
}

final class _StrictJsonParser {
  _StrictJsonParser(this.source);

  final String source;
  int index = 0;

  Object? parseDocument() {
    _skipWhitespace();
    final value = _parseValue(0);
    if (index != source.length) {
      throw const ContractFailure(
        ContractFailureCode.invalidJson,
        'trailing bytes after JSON value',
      );
    }
    return value;
  }

  Object? _parseValue(int depth) {
    _requireSafeDepth(depth);
    if (index >= source.length) return _invalidJson('unexpected end of JSON');
    return switch (source.codeUnitAt(index)) {
      0x7b => _parseObject(depth),
      0x5b => _parseArray(depth),
      0x22 => _parseString(),
      0x74 => _parseLiteral('true', true),
      0x66 => _parseLiteral('false', false),
      0x6e => _parseLiteral('null', null),
      final unit when unit == 0x2d || (unit >= 0x30 && unit <= 0x39) =>
        _parseInteger(),
      _ => _invalidJson('unexpected JSON token'),
    };
  }

  Map<String, Object?> _parseObject(int depth) {
    index += 1;
    final result = <String, Object?>{};
    _skipWhitespace();
    if (_consume(0x7d)) return result;
    while (true) {
      if (index >= source.length || source.codeUnitAt(index) != 0x22) {
        return _invalidJson('object key must be a string');
      }
      final key = _parseString();
      if (result.containsKey(key)) {
        throw ContractFailure(
          ContractFailureCode.duplicateKey,
          'duplicate object key: $key',
        );
      }
      _skipWhitespace();
      _require(0x3a, 'missing colon after object key');
      _skipWhitespace();
      result[key] = _parseValue(depth + 1);
      _skipWhitespace();
      if (_consume(0x7d)) return result;
      _require(0x2c, 'missing comma between object fields');
      _skipWhitespace();
    }
  }

  List<Object?> _parseArray(int depth) {
    index += 1;
    final result = <Object?>[];
    _skipWhitespace();
    if (_consume(0x5d)) return result;
    while (true) {
      result.add(_parseValue(depth + 1));
      _skipWhitespace();
      if (_consume(0x5d)) return result;
      _require(0x2c, 'missing comma between array items');
      _skipWhitespace();
    }
  }

  String _parseString() {
    index += 1;
    final result = StringBuffer();
    while (index < source.length) {
      final unit = source.codeUnitAt(index);
      if (unit == 0x22) {
        index += 1;
        return result.toString();
      }
      if (unit < 0x20) {
        return _invalidJson('control character in string');
      }
      if (unit == 0x5c) {
        index += 1;
        _parseStringEscape(result);
        continue;
      }
      if (unit >= 0xd800 && unit <= 0xdbff) {
        if (index + 1 >= source.length) {
          return _invalidJson('unpaired high surrogate in string');
        }
        final low = source.codeUnitAt(index + 1);
        if (low < 0xdc00 || low > 0xdfff) {
          return _invalidJson('unpaired high surrogate in string');
        }
        result.writeCharCode(0x10000 + ((unit - 0xd800) << 10) + low - 0xdc00);
        index += 2;
        continue;
      }
      if (unit >= 0xdc00 && unit <= 0xdfff) {
        return _invalidJson('unpaired low surrogate in string');
      }
      result.writeCharCode(unit);
      index += 1;
    }
    return _invalidJson('unterminated JSON string');
  }

  void _parseStringEscape(StringBuffer result) {
    if (index >= source.length) _invalidJson('incomplete string escape');
    final escape = source.codeUnitAt(index);
    index += 1;
    switch (escape) {
      case 0x22:
      case 0x2f:
      case 0x5c:
        result.writeCharCode(escape);
      case 0x62:
        result.writeCharCode(0x08);
      case 0x66:
        result.writeCharCode(0x0c);
      case 0x6e:
        result.writeCharCode(0x0a);
      case 0x72:
        result.writeCharCode(0x0d);
      case 0x74:
        result.writeCharCode(0x09);
      case 0x75:
        final first = _parseUnicodeEscape();
        if (first >= 0xd800 && first <= 0xdbff) {
          if (index + 1 >= source.length ||
              source.codeUnitAt(index) != 0x5c ||
              source.codeUnitAt(index + 1) != 0x75) {
            _invalidJson('unpaired high surrogate escape');
          }
          index += 2;
          final low = _parseUnicodeEscape();
          if (low < 0xdc00 || low > 0xdfff) {
            _invalidJson('invalid low surrogate escape');
          }
          result.writeCharCode(
            0x10000 + ((first - 0xd800) << 10) + low - 0xdc00,
          );
        } else if (first >= 0xdc00 && first <= 0xdfff) {
          _invalidJson('unpaired low surrogate escape');
        } else {
          result.writeCharCode(first);
        }
      default:
        _invalidJson('invalid string escape');
    }
  }

  int _parseUnicodeEscape() {
    if (index + 4 > source.length) _invalidJson('incomplete Unicode escape');
    final digits = source.substring(index, index + 4);
    final value = int.tryParse(digits, radix: 16);
    if (value == null) _invalidJson('invalid Unicode escape');
    index += 4;
    return value;
  }

  Object? _parseLiteral(String token, Object? value) {
    if (!source.startsWith(token, index)) {
      return _invalidJson('invalid literal');
    }
    index += token.length;
    return value;
  }

  BigInt _parseInteger() {
    final start = index;
    if (_consume(0x2d) && index >= source.length) {
      return _invalidJson('incomplete number');
    }
    if (_consume(0x30)) {
      if (index < source.length && _isDigit(source.codeUnitAt(index))) {
        return _invalidNumber('leading zero is not canonical');
      }
    } else {
      if (index >= source.length ||
          !_isDigitOneToNine(source.codeUnitAt(index))) {
        return _invalidJson('invalid number');
      }
      while (index < source.length && _isDigit(source.codeUnitAt(index))) {
        index += 1;
      }
    }
    if (index < source.length) {
      final unit = source.codeUnitAt(index);
      if (unit == 0x2e || unit == 0x65 || unit == 0x45) {
        return _invalidNumber('only strict integer numbers are accepted');
      }
    }
    return BigInt.parse(source.substring(start, index));
  }

  void _skipWhitespace() {
    while (index < source.length) {
      final unit = source.codeUnitAt(index);
      if (unit != 0x20 && unit != 0x0a && unit != 0x0d && unit != 0x09) return;
      index += 1;
    }
  }

  bool _consume(int unit) {
    if (index < source.length && source.codeUnitAt(index) == unit) {
      index += 1;
      return true;
    }
    return false;
  }

  void _require(int unit, String message) {
    if (!_consume(unit)) _invalidJson(message);
  }

  bool _isDigit(int unit) => unit >= 0x30 && unit <= 0x39;
  bool _isDigitOneToNine(int unit) => unit >= 0x31 && unit <= 0x39;

  Never _invalidJson(String message) =>
      throw ContractFailure(ContractFailureCode.invalidJson, message);

  Never _invalidNumber(String message) =>
      throw ContractFailure(ContractFailureCode.invalidNumber, message);
}
