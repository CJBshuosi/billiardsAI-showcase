import 'dart:convert';
import 'dart:io';

import 'package:crypto/crypto.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  test('reviewed UI-state V2 fixtures preserve exact canonical bytes', () {
    final fixtures = {
      'post_shot_success.json': (
        length: 2113,
        digest:
            'a90a29dcd7602d8f549ea36de268c3925435a371fdee9cf8caaa3085a6440bc4',
      ),
      'abstained.json': (
        length: 1558,
        digest:
            'c1d5e0d07f329ab367b3092168a375bae46b06fa666b5f19afa3407cbbb7489c',
      ),
      'unsupported.json': (
        length: 1562,
        digest:
            'cc3f2a1b3b6d8576c66d952472335703031a4d70cd0f7106c19d87e0e7834684',
      ),
      'degraded.json': (
        length: 1549,
        digest:
            '0fa158834aad0a373782a0e2b4a4480133e0b47380a0d13d73746158d2c2d22a',
      ),
    };

    for (final MapEntry(key: name, value: expected) in fixtures.entries) {
      final bytes = File('test/fixtures/realtime_v2/$name').readAsBytesSync();
      expect(bytes, hasLength(expected.length), reason: name);
      expect(bytes.last, isNot(0x0a), reason: name);
      expect(sha256.convert(bytes).toString(), expected.digest, reason: name);
      final decoded = jsonDecode(utf8.decode(bytes));
      expect(decoded, isA<Map<String, Object?>>(), reason: name);
      expect(
        (decoded as Map<String, Object?>)['schema_version'],
        '2.0',
        reason: name,
      );
    }
  });

  test('provenance generation is frozen to the archived uv project', () {
    final provenance = File(
      'test/fixtures/realtime_v2/PROVENANCE.md',
    ).readAsStringSync();

    expect(provenance, contains(r'APP_REPO_ROOT="$(pwd -P)"'));
    expect(
      provenance,
      contains(r'uv sync --project "$UPSTREAM_DIR" --frozen --extra dev'),
    );
    expect(
      provenance,
      contains(r'uv run --project "$UPSTREAM_DIR" --frozen python'),
    );
    expect(
      provenance,
      contains(r'"$APP_REPO_ROOT/apps/mobile/test/fixtures/realtime_v2"'),
    );
  });

  test('verified root events are decoder-only across production and tests', () {
    final contract = File(
      'lib/core/contracts/realtime_coaching_event_v2.dart',
    ).readAsStringSync();
    expect(contract, contains('RealtimeCoachingEventV2._decoded('));
    expect(contract, isNot(contains('const RealtimeCoachingEventV2({')));

    final sourceRoots = [Directory('lib'), Directory('test')];
    final directConstruction = RegExp(r'\bRealtimeCoachingEventV2\s*\(');
    final offenders = <String>[];
    for (final root in sourceRoots) {
      for (final entity in root.listSync(recursive: true)) {
        if (entity is! File || !entity.path.endsWith('.dart')) continue;
        if (entity.path.endsWith('realtime_fixture_provenance_test.dart')) {
          continue;
        }
        if (directConstruction.hasMatch(entity.readAsStringSync())) {
          offenders.add(entity.path);
        }
      }
    }
    expect(offenders, isEmpty);
  });

  test('all replay vectors preserve exact authority bytes', () {
    final vectors = {
      'conflicting_sequence_9.json': (
        length: 2099,
        digest:
            '98dcd9f867b022695d73a98182bac881459df61864a2de22ba206184dca72366',
      ),
      'sequence_10.json': (
        length: 2114,
        digest:
            '702095ca7083a58196ebc14a07ec0898e11dae7e735b0ff6f4282ca370f3774c',
      ),
      'sequence_11.json': (
        length: 2114,
        digest:
            '11378d9c3b1bbc89bd253e8baf5a77d590eb3abf84356f638752b76f25dc0b29',
      ),
    };

    for (final MapEntry(key: name, value: expected) in vectors.entries) {
      final bytes = File('test/fixtures/replay_v2/$name').readAsBytesSync();
      expect(bytes, hasLength(expected.length), reason: name);
      expect(bytes.last, isNot(0x0a), reason: name);
      expect(sha256.convert(bytes).toString(), expected.digest, reason: name);
    }
  });

  test('replay provenance freezes one executable archive invocation', () {
    final provenance = File(
      'test/fixtures/replay_v2/PROVENANCE.md',
    ).readAsStringSync();

    expect(provenance, contains('6825068e85e4eecfe89a19b8a8854fb4c2f454dd'));
    expect(provenance, contains(r'git archive "$UPSTREAM_COMMIT"'));
    expect(
      provenance,
      contains(r'uv sync --project "$UPSTREAM_DIR" --frozen --extra dev'),
    );
    expect(
      provenance,
      contains(r'uv run --project "$UPSTREAM_DIR" --frozen python'),
    );
    for (final name in [
      'conflicting_sequence_9.json',
      'sequence_10.json',
      'sequence_11.json',
    ]) {
      expect(provenance, contains(name));
    }
  });
}
