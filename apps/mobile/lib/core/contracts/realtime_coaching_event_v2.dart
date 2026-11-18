import 'dart:convert';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';

import 'canonical_json.dart';
import 'contract_failure.dart';

final BigInt _tableLengthMm = BigInt.from(2540);
final BigInt _tableWidthMm = BigInt.from(1270);

enum CoachingEventType {
  setup,
  preShotCue,
  postShotFeedback,
  progression,
  regression,
  abstention,
  restSuggestion,
  sessionSummary,
  systemStatus,
}

enum CoachingOutcomeStatus {
  informational,
  success,
  failure,
  abstained,
  unsupported,
}

enum CoachingDeliveryStatus { normal, degraded, error }

enum OverlayCoordinateSpace { tableMm }

enum OverlayPrimitiveKind { point, zone }

enum OverlaySemanticRole { actualCueBallEnd, status, targetZone }

enum OverlayLineStyle { solid, dashed, dotted }

final class TablePointMmV1 {
  const TablePointMmV1({
    required this.schemaVersion,
    required this.xMm,
    required this.yMm,
  });

  final String schemaVersion;
  final BigInt xMm;
  final BigInt yMm;
}

final class TableZoneMmV1 {
  const TableZoneMmV1({
    required this.schemaVersion,
    required this.minimumXMm,
    required this.maximumXMm,
    required this.minimumYMm,
    required this.maximumYMm,
  });

  final String schemaVersion;
  final BigInt minimumXMm;
  final BigInt maximumXMm;
  final BigInt minimumYMm;
  final BigInt maximumYMm;
}

final class OverlayPrimitiveV1 {
  const OverlayPrimitiveV1({
    required this.schemaVersion,
    required this.kind,
    required this.semanticRole,
    required this.labelCode,
    required this.lineStyle,
    required this.point,
    required this.zone,
  });

  final String schemaVersion;
  final OverlayPrimitiveKind kind;
  final OverlaySemanticRole semanticRole;
  final String labelCode;
  final OverlayLineStyle lineStyle;
  final TablePointMmV1? point;
  final TableZoneMmV1? zone;
}

final class OverlayPayloadV1 {
  OverlayPayloadV1({
    required this.schemaVersion,
    required this.coordinateSpace,
    required this.outcomeStatus,
    required this.deliveryStatus,
    required List<OverlayPrimitiveV1> primitives,
  }) : primitives = List.unmodifiable(primitives);

  final String schemaVersion;
  final OverlayCoordinateSpace coordinateSpace;
  final CoachingOutcomeStatus outcomeStatus;
  final CoachingDeliveryStatus deliveryStatus;
  final List<OverlayPrimitiveV1> primitives;
}

final class SpeechCueV1 {
  const SpeechCueV1({
    required this.schemaVersion,
    required this.locale,
    required this.cueCode,
    required this.outcomeStatus,
    required this.deliveryStatus,
    required this.text,
  });

  final String schemaVersion;
  final String locale;
  final String cueCode;
  final CoachingOutcomeStatus outcomeStatus;
  final CoachingDeliveryStatus deliveryStatus;
  final String text;
}

final class RealtimeCoachingLineageV1 {
  const RealtimeCoachingLineageV1({
    required this.schemaVersion,
    required this.admissionId,
    required this.bundleId,
    required this.bundleContentSha256,
    required this.projectionSha256,
    required this.sessionId,
    required this.planId,
    required this.planVersion,
    required this.blockId,
    required this.blockOrder,
    required this.drillId,
    required this.drillRevision,
    required this.drillContentSha256,
    required this.shotId,
    required this.shotSequence,
    required this.evidenceSha256,
    required this.decisionId,
    required this.effectId,
  });

  final String schemaVersion;
  final String admissionId;
  final String bundleId;
  final String bundleContentSha256;
  final String projectionSha256;
  final String sessionId;
  final String planId;
  final BigInt planVersion;
  final String blockId;
  final BigInt blockOrder;
  final String drillId;
  final BigInt drillRevision;
  final String drillContentSha256;
  final String? shotId;
  final BigInt? shotSequence;
  final String? evidenceSha256;
  final String? decisionId;
  final String effectId;
}

final class RealtimeCoachingEventV2 {
  const RealtimeCoachingEventV2._decoded({
    required this.schemaVersion,
    required this.eventId,
    required this.sequence,
    required this.eventType,
    required this.outcomeStatus,
    required this.deliveryStatus,
    required this.lineage,
    required this.overlay,
    required this.speech,
    required this.textSummary,
    required this.createdMonotonicMs,
    required this.idempotencyKey,
    required this.contentSha256,
  });

  final String schemaVersion;
  final String eventId;
  final BigInt sequence;
  final CoachingEventType eventType;
  final CoachingOutcomeStatus outcomeStatus;
  final CoachingDeliveryStatus deliveryStatus;
  final RealtimeCoachingLineageV1 lineage;
  final OverlayPayloadV1? overlay;
  final SpeechCueV1? speech;
  final String textSummary;
  final BigInt createdMonotonicMs;
  final String idempotencyKey;
  final String contentSha256;
}

RealtimeCoachingEventV2 decodeRealtimeCoachingEventV2(Uint8List bytes) {
  final decoded = parseCanonicalJson(bytes);
  final root = _object(decoded, 'event');
  _exactKeys(root, const {
    'schema_version',
    'event_id',
    'sequence',
    'event_type',
    'outcome_status',
    'delivery_status',
    'lineage',
    'overlay',
    'speech',
    'text_summary',
    'created_monotonic_ms',
    'idempotency_key',
    'content_sha256',
  }, 'event');

  final schemaVersion = _string(root['schema_version'], 'event.schema_version');
  if (schemaVersion != '2.0') {
    throw const ContractFailure(
      ContractFailureCode.unsupportedSchemaVersion,
      'only realtime coaching event schema 2.0 is supported',
    );
  }
  final eventId = _uuid4(root['event_id'], 'event.event_id');
  final sequence = _positiveInt(root['sequence'], 'event.sequence');
  final eventType = _eventType(root['event_type']);
  final outcomeStatus = _outcomeStatus(
    root['outcome_status'],
    'event.outcome_status',
  );
  final deliveryStatus = _deliveryStatus(
    root['delivery_status'],
    'event.delivery_status',
  );
  final lineageMap = _object(root['lineage'], 'event.lineage');
  final lineage = _parseLineage(lineageMap);
  final overlay = root['overlay'] == null
      ? null
      : _parseOverlay(_object(root['overlay'], 'event.overlay'));
  final speech = root['speech'] == null
      ? null
      : _parseSpeech(_object(root['speech'], 'event.speech'));
  final textSummary = _nonEmpty(root['text_summary'], 'event.text_summary');
  final createdMonotonicMs = _nonNegativeInt(
    root['created_monotonic_ms'],
    'event.created_monotonic_ms',
  );
  final idempotencyKey = _digest(
    root['idempotency_key'],
    'event.idempotency_key',
  );
  final contentSha256 = _digest(root['content_sha256'], 'event.content_sha256');

  _validateStatuses(
    eventType: eventType,
    outcomeStatus: outcomeStatus,
    deliveryStatus: deliveryStatus,
    lineage: lineage,
    overlay: overlay,
    speech: speech,
    textSummary: textSummary,
  );
  _validateHashes(
    root: root,
    eventId: eventId,
    idempotencyKey: idempotencyKey,
    contentSha256: contentSha256,
  );

  return RealtimeCoachingEventV2._decoded(
    schemaVersion: schemaVersion,
    eventId: eventId,
    sequence: sequence,
    eventType: eventType,
    outcomeStatus: outcomeStatus,
    deliveryStatus: deliveryStatus,
    lineage: lineage,
    overlay: overlay,
    speech: speech,
    textSummary: textSummary,
    createdMonotonicMs: createdMonotonicMs,
    idempotencyKey: idempotencyKey,
    contentSha256: contentSha256,
  );
}

RealtimeCoachingLineageV1 _parseLineage(Map<String, Object?> value) {
  _exactKeys(value, const {
    'schema_version',
    'admission_id',
    'bundle_id',
    'bundle_content_sha256',
    'projection_sha256',
    'session_id',
    'plan_id',
    'plan_version',
    'block_id',
    'block_order',
    'drill_id',
    'drill_revision',
    'drill_content_sha256',
    'shot_id',
    'shot_sequence',
    'evidence_sha256',
    'decision_id',
    'effect_id',
  }, 'lineage');
  _schemaV1(value['schema_version'], 'lineage.schema_version');
  final shotId = _nullableUuid4(value['shot_id'], 'lineage.shot_id');
  final shotSequence = _nullablePositiveInt(
    value['shot_sequence'],
    'lineage.shot_sequence',
  );
  final evidenceSha256 = _nullableDigest(
    value['evidence_sha256'],
    'lineage.evidence_sha256',
  );
  final decisionId = _nullableDigest(
    value['decision_id'],
    'lineage.decision_id',
  );
  final shotParts = <Object?>[shotId, shotSequence, evidenceSha256];
  if (shotParts.any((part) => part != null) &&
      shotParts.any((part) => part == null)) {
    throw const ContractFailure(
      ContractFailureCode.lineageMismatch,
      'shot lineage must be wholly present or absent',
    );
  }
  if (decisionId != null && shotId == null) {
    throw const ContractFailure(
      ContractFailureCode.lineageMismatch,
      'decision lineage requires shot lineage',
    );
  }
  return RealtimeCoachingLineageV1(
    schemaVersion: '1.0',
    admissionId: _nonEmpty(value['admission_id'], 'lineage.admission_id'),
    bundleId: _uuid4(value['bundle_id'], 'lineage.bundle_id'),
    bundleContentSha256: _digest(
      value['bundle_content_sha256'],
      'lineage.bundle_content_sha256',
    ),
    projectionSha256: _digest(
      value['projection_sha256'],
      'lineage.projection_sha256',
    ),
    sessionId: _uuid4(value['session_id'], 'lineage.session_id'),
    planId: _uuid4(value['plan_id'], 'lineage.plan_id'),
    planVersion: _positiveInt(value['plan_version'], 'lineage.plan_version'),
    blockId: _uuid4(value['block_id'], 'lineage.block_id'),
    blockOrder: _positiveInt(value['block_order'], 'lineage.block_order'),
    drillId: _nonEmpty(value['drill_id'], 'lineage.drill_id'),
    drillRevision: _positiveInt(
      value['drill_revision'],
      'lineage.drill_revision',
    ),
    drillContentSha256: _digest(
      value['drill_content_sha256'],
      'lineage.drill_content_sha256',
    ),
    shotId: shotId,
    shotSequence: shotSequence,
    evidenceSha256: evidenceSha256,
    decisionId: decisionId,
    effectId: _digest(value['effect_id'], 'lineage.effect_id'),
  );
}

OverlayPayloadV1 _parseOverlay(Map<String, Object?> value) {
  _exactKeys(value, const {
    'schema_version',
    'coordinate_space',
    'outcome_status',
    'delivery_status',
    'primitives',
  }, 'overlay');
  _schemaV1(value['schema_version'], 'overlay.schema_version');
  if (_string(value['coordinate_space'], 'overlay.coordinate_space') !=
      'TABLE_MM') {
    _invalidValue('overlay.coordinate_space must be TABLE_MM');
  }
  final rawPrimitives = _list(value['primitives'], 'overlay.primitives');
  final primitives = <OverlayPrimitiveV1>[];
  for (var index = 0; index < rawPrimitives.length; index += 1) {
    primitives.add(
      _parsePrimitive(
        _object(rawPrimitives[index], 'overlay.primitives[$index]'),
      ),
    );
  }
  final roles = primitives
      .map((primitive) => _roleWire(primitive.semanticRole))
      .toList();
  if (roles.toSet().length != roles.length) {
    _overlayFailure('overlay semantic roles must be unique');
  }
  final sortedRoles = [...roles]..sort();
  if (!_stringListsEqual(roles, sortedRoles)) {
    _overlayFailure(
      'overlay primitives must use canonical semantic-role order',
    );
  }
  return OverlayPayloadV1(
    schemaVersion: '1.0',
    coordinateSpace: OverlayCoordinateSpace.tableMm,
    outcomeStatus: _outcomeStatus(
      value['outcome_status'],
      'overlay.outcome_status',
    ),
    deliveryStatus: _deliveryStatus(
      value['delivery_status'],
      'overlay.delivery_status',
    ),
    primitives: primitives,
  );
}

OverlayPrimitiveV1 _parsePrimitive(Map<String, Object?> value) {
  _exactKeys(value, const {
    'schema_version',
    'kind',
    'semantic_role',
    'label_code',
    'line_style',
    'point',
    'zone',
  }, 'overlay primitive');
  _schemaV1(value['schema_version'], 'overlay primitive.schema_version');
  final kind = _primitiveKind(value['kind']);
  final role = _semanticRole(value['semantic_role']);
  final point = value['point'] == null
      ? null
      : _parsePoint(_object(value['point'], 'overlay primitive.point'));
  final zone = value['zone'] == null
      ? null
      : _parseZone(_object(value['zone'], 'overlay primitive.zone'));
  if (kind == OverlayPrimitiveKind.point && (point == null || zone != null)) {
    _overlayFailure('POINT overlay requires only point geometry');
  }
  if (kind == OverlayPrimitiveKind.zone && (zone == null || point != null)) {
    _overlayFailure('ZONE overlay requires only zone geometry');
  }
  if (role == OverlaySemanticRole.status) {
    _overlayFailure('STATUS has no authoritative geometry in overlay v1');
  }
  final expectedKind = role == OverlaySemanticRole.actualCueBallEnd
      ? OverlayPrimitiveKind.point
      : OverlayPrimitiveKind.zone;
  if (kind != expectedKind) {
    _overlayFailure('overlay semantic role does not match geometry kind');
  }
  return OverlayPrimitiveV1(
    schemaVersion: '1.0',
    kind: kind,
    semanticRole: role,
    labelCode: _nonEmpty(value['label_code'], 'overlay primitive.label_code'),
    lineStyle: _lineStyle(value['line_style']),
    point: point,
    zone: zone,
  );
}

TablePointMmV1 _parsePoint(Map<String, Object?> value) {
  _exactKeys(value, const {'schema_version', 'x_mm', 'y_mm'}, 'table point');
  _schemaV1(value['schema_version'], 'table point.schema_version');
  final x = _int(value['x_mm'], 'table point.x_mm');
  final y = _int(value['y_mm'], 'table point.y_mm');
  if (x < BigInt.zero ||
      x > _tableLengthMm ||
      y < BigInt.zero ||
      y > _tableWidthMm) {
    _overlayFailure('table point is outside authoritative millimetre bounds');
  }
  return TablePointMmV1(schemaVersion: '1.0', xMm: x, yMm: y);
}

TableZoneMmV1 _parseZone(Map<String, Object?> value) {
  _exactKeys(value, const {
    'schema_version',
    'minimum_x_mm',
    'maximum_x_mm',
    'minimum_y_mm',
    'maximum_y_mm',
  }, 'table zone');
  _schemaV1(value['schema_version'], 'table zone.schema_version');
  final minimumX = _int(value['minimum_x_mm'], 'table zone.minimum_x_mm');
  final maximumX = _int(value['maximum_x_mm'], 'table zone.maximum_x_mm');
  final minimumY = _int(value['minimum_y_mm'], 'table zone.minimum_y_mm');
  final maximumY = _int(value['maximum_y_mm'], 'table zone.maximum_y_mm');
  if (minimumX > maximumX || minimumY > maximumY) {
    _overlayFailure('table zone bounds must be ordered');
  }
  if (minimumX < BigInt.zero ||
      maximumX > _tableLengthMm ||
      minimumY < BigInt.zero ||
      maximumY > _tableWidthMm) {
    _overlayFailure('table zone is outside authoritative millimetre bounds');
  }
  return TableZoneMmV1(
    schemaVersion: '1.0',
    minimumXMm: minimumX,
    maximumXMm: maximumX,
    minimumYMm: minimumY,
    maximumYMm: maximumY,
  );
}

SpeechCueV1 _parseSpeech(Map<String, Object?> value) {
  _exactKeys(value, const {
    'schema_version',
    'locale',
    'cue_code',
    'outcome_status',
    'delivery_status',
    'text',
  }, 'speech');
  _schemaV1(value['schema_version'], 'speech.schema_version');
  final locale = _string(value['locale'], 'speech.locale');
  if (locale != 'zh-CN') _invalidValue('speech.locale must be zh-CN');
  return SpeechCueV1(
    schemaVersion: '1.0',
    locale: locale,
    cueCode: _nonEmpty(value['cue_code'], 'speech.cue_code'),
    outcomeStatus: _outcomeStatus(
      value['outcome_status'],
      'speech.outcome_status',
    ),
    deliveryStatus: _deliveryStatus(
      value['delivery_status'],
      'speech.delivery_status',
    ),
    text: _nonEmpty(value['text'], 'speech.text'),
  );
}

void _validateStatuses({
  required CoachingEventType eventType,
  required CoachingOutcomeStatus outcomeStatus,
  required CoachingDeliveryStatus deliveryStatus,
  required RealtimeCoachingLineageV1 lineage,
  required OverlayPayloadV1? overlay,
  required SpeechCueV1? speech,
  required String textSummary,
}) {
  if (overlay != null &&
      (overlay.outcomeStatus != outcomeStatus ||
          overlay.deliveryStatus != deliveryStatus)) {
    _statusFailure('overlay status must match event status');
  }
  if (speech != null &&
      (speech.outcomeStatus != outcomeStatus ||
          speech.deliveryStatus != deliveryStatus)) {
    _statusFailure('speech status must match event status');
  }
  if (speech != null && speech.text != textSummary) {
    _statusFailure('speech text must match event text summary');
  }
  if ((deliveryStatus != CoachingDeliveryStatus.normal ||
          outcomeStatus == CoachingOutcomeStatus.abstained ||
          outcomeStatus == CoachingOutcomeStatus.unsupported) &&
      speech == null) {
    _statusFailure(
      'abstention, degraded, and error events require a speech cue',
    );
  }
  final combination =
      '${_eventTypeWire(eventType)}|${_outcomeWire(outcomeStatus)}|${_deliveryWire(deliveryStatus)}';
  const allowed = {
    'SETUP|INFORMATIONAL|NORMAL',
    'PRE_SHOT_CUE|INFORMATIONAL|NORMAL',
    'POST_SHOT_FEEDBACK|SUCCESS|NORMAL',
    'POST_SHOT_FEEDBACK|FAILURE|NORMAL',
    'PROGRESSION|SUCCESS|NORMAL',
    'REGRESSION|FAILURE|NORMAL',
    'ABSTENTION|ABSTAINED|NORMAL',
    'ABSTENTION|UNSUPPORTED|NORMAL',
    'REST_SUGGESTION|INFORMATIONAL|NORMAL',
    'SESSION_SUMMARY|INFORMATIONAL|NORMAL',
    'SYSTEM_STATUS|INFORMATIONAL|NORMAL',
    'SYSTEM_STATUS|INFORMATIONAL|DEGRADED',
    'SYSTEM_STATUS|INFORMATIONAL|ERROR',
    'SYSTEM_STATUS|ABSTAINED|DEGRADED',
    'SYSTEM_STATUS|UNSUPPORTED|DEGRADED',
  };
  if (!allowed.contains(combination)) {
    _statusFailure(
      'event type, outcome, and delivery statuses are inconsistent',
    );
  }
  if (outcomeStatus != CoachingOutcomeStatus.informational &&
      (lineage.shotId == null || lineage.decisionId == null)) {
    throw const ContractFailure(
      ContractFailureCode.lineageMismatch,
      'machine outcome requires shot and decision lineage',
    );
  }
}

void _validateHashes({
  required Map<String, Object?> root,
  required String eventId,
  required String idempotencyKey,
  required String contentSha256,
}) {
  final identity = <String, Object?>{
    'schema_version': root['schema_version'],
    'sequence': root['sequence'],
    'event_type': root['event_type'],
    'outcome_status': root['outcome_status'],
    'delivery_status': root['delivery_status'],
    'lineage': root['lineage'],
    'created_monotonic_ms': root['created_monotonic_ms'],
  };
  final identityDigest = _domainSha256(
    'realtime-coaching-event-identity/v2',
    identity,
  );
  if (eventId != _uuid4FromDigest(identityDigest)) {
    _hashFailure('event identity mismatch');
  }
  final expectedIdempotency = _domainSha256(
    'realtime-coaching-event-idempotency/v2',
    identity,
  );
  if (idempotencyKey != expectedIdempotency) {
    _hashFailure('event idempotency identity mismatch');
  }
  final content = <String, Object?>{
    for (final entry in root.entries)
      if (entry.key != 'content_sha256') entry.key: entry.value,
  };
  final expectedContent = _domainSha256(
    'realtime-coaching-event-content/v2',
    content,
  );
  if (contentSha256 != expectedContent) {
    _hashFailure('event content hash mismatch');
  }
}

String _domainSha256(String domain, Object? material) {
  final bytes = <int>[
    ...utf8.encode(domain),
    0,
    ...canonicalJsonBytes(material),
  ];
  return sha256.convert(bytes).toString();
}

String _uuid4FromDigest(String digest) {
  final bytes = <int>[
    for (var offset = 0; offset < 32; offset += 2)
      int.parse(digest.substring(offset, offset + 2), radix: 16),
  ];
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  final hex = bytes
      .map((byte) => byte.toRadixString(16).padLeft(2, '0'))
      .join();
  return '${hex.substring(0, 8)}-${hex.substring(8, 12)}-${hex.substring(12, 16)}-${hex.substring(16, 20)}-${hex.substring(20, 32)}';
}

Map<String, Object?> _object(Object? value, String path) {
  if (value is Map<String, Object?>) return value;
  _invalidValue('$path must be an object');
}

List<Object?> _list(Object? value, String path) {
  if (value is List<Object?>) return value;
  _invalidValue('$path must be an array');
}

String _string(Object? value, String path) {
  if (value is String) return value;
  _invalidValue('$path must be a string');
}

String _nonEmpty(Object? value, String path) {
  final result = _string(value, path);
  final stripped = _pythonStrip(result);
  if (stripped.isEmpty || stripped != result) {
    _invalidValue('$path must be a trimmed non-empty string');
  }
  return result;
}

String _pythonStrip(String value) {
  final codePoints = value.runes.toList(growable: false);
  var start = 0;
  while (start < codePoints.length && _isPythonWhitespace(codePoints[start])) {
    start += 1;
  }
  var end = codePoints.length;
  while (end > start && _isPythonWhitespace(codePoints[end - 1])) {
    end -= 1;
  }
  return String.fromCharCodes(codePoints.sublist(start, end));
}

bool _isPythonWhitespace(int codePoint) =>
    (codePoint >= 0x09 && codePoint <= 0x0d) ||
    (codePoint >= 0x1c && codePoint <= 0x20) ||
    codePoint == 0x85 ||
    codePoint == 0xa0 ||
    codePoint == 0x1680 ||
    (codePoint >= 0x2000 && codePoint <= 0x200a) ||
    codePoint == 0x2028 ||
    codePoint == 0x2029 ||
    codePoint == 0x202f ||
    codePoint == 0x205f ||
    codePoint == 0x3000;

BigInt _int(Object? value, String path) {
  if (value is BigInt) return value;
  _invalidValue('$path must be a strict integer');
}

BigInt _positiveInt(Object? value, String path) {
  final result = _int(value, path);
  if (result < BigInt.one) _invalidValue('$path must be at least 1');
  return result;
}

BigInt _nonNegativeInt(Object? value, String path) {
  final result = _int(value, path);
  if (result < BigInt.zero) _invalidValue('$path must be non-negative');
  return result;
}

BigInt? _nullablePositiveInt(Object? value, String path) =>
    value == null ? null : _positiveInt(value, path);

String _uuid4(Object? value, String path) {
  final result = _string(value, path);
  final pattern = RegExp(
    r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$',
  );
  if (!pattern.hasMatch(result)) {
    throw ContractFailure(
      ContractFailureCode.invalidUuid,
      '$path must be canonical UUIDv4',
    );
  }
  return result;
}

String? _nullableUuid4(Object? value, String path) =>
    value == null ? null : _uuid4(value, path);

String _digest(Object? value, String path) {
  final result = _string(value, path);
  if (!RegExp(r'^[0-9a-f]{64}$').hasMatch(result)) {
    throw ContractFailure(
      ContractFailureCode.invalidDigest,
      '$path must be lowercase SHA-256',
    );
  }
  return result;
}

String? _nullableDigest(Object? value, String path) =>
    value == null ? null : _digest(value, path);

void _schemaV1(Object? value, String path) {
  if (_string(value, path) != '1.0') _invalidValue('$path must be 1.0');
}

void _exactKeys(Map<String, Object?> value, Set<String> expected, String path) {
  final unknown = value.keys.where((key) => !expected.contains(key)).toList()
    ..sort();
  if (unknown.isNotEmpty) {
    throw ContractFailure(
      ContractFailureCode.unknownField,
      '$path contains unknown field ${unknown.first}',
    );
  }
  final missing = expected.where((key) => !value.containsKey(key)).toList()
    ..sort();
  if (missing.isNotEmpty) {
    throw ContractFailure(
      ContractFailureCode.missingField,
      '$path is missing required field ${missing.first}',
    );
  }
}

CoachingEventType _eventType(Object? value) =>
    switch (_string(value, 'event.event_type')) {
      'SETUP' => CoachingEventType.setup,
      'PRE_SHOT_CUE' => CoachingEventType.preShotCue,
      'POST_SHOT_FEEDBACK' => CoachingEventType.postShotFeedback,
      'PROGRESSION' => CoachingEventType.progression,
      'REGRESSION' => CoachingEventType.regression,
      'ABSTENTION' => CoachingEventType.abstention,
      'REST_SUGGESTION' => CoachingEventType.restSuggestion,
      'SESSION_SUMMARY' => CoachingEventType.sessionSummary,
      'SYSTEM_STATUS' => CoachingEventType.systemStatus,
      _ => _invalidValue('unknown event type'),
    };

CoachingOutcomeStatus _outcomeStatus(Object? value, String path) =>
    switch (_string(value, path)) {
      'INFORMATIONAL' => CoachingOutcomeStatus.informational,
      'SUCCESS' => CoachingOutcomeStatus.success,
      'FAILURE' => CoachingOutcomeStatus.failure,
      'ABSTAINED' => CoachingOutcomeStatus.abstained,
      'UNSUPPORTED' => CoachingOutcomeStatus.unsupported,
      _ => _invalidValue('unknown outcome status at $path'),
    };

CoachingDeliveryStatus _deliveryStatus(Object? value, String path) =>
    switch (_string(value, path)) {
      'NORMAL' => CoachingDeliveryStatus.normal,
      'DEGRADED' => CoachingDeliveryStatus.degraded,
      'ERROR' => CoachingDeliveryStatus.error,
      _ => _invalidValue('unknown delivery status at $path'),
    };

OverlayPrimitiveKind _primitiveKind(Object? value) =>
    switch (_string(value, 'overlay primitive.kind')) {
      'POINT' => OverlayPrimitiveKind.point,
      'ZONE' => OverlayPrimitiveKind.zone,
      _ => _invalidValue('unknown overlay primitive kind'),
    };

OverlaySemanticRole _semanticRole(Object? value) =>
    switch (_string(value, 'overlay primitive.semantic_role')) {
      'ACTUAL_CUE_BALL_END' => OverlaySemanticRole.actualCueBallEnd,
      'STATUS' => OverlaySemanticRole.status,
      'TARGET_ZONE' => OverlaySemanticRole.targetZone,
      _ => _invalidValue('unknown overlay semantic role'),
    };

OverlayLineStyle _lineStyle(Object? value) =>
    switch (_string(value, 'overlay primitive.line_style')) {
      'SOLID' => OverlayLineStyle.solid,
      'DASHED' => OverlayLineStyle.dashed,
      'DOTTED' => OverlayLineStyle.dotted,
      _ => _invalidValue('unknown overlay line style'),
    };

String _eventTypeWire(CoachingEventType value) => switch (value) {
  CoachingEventType.setup => 'SETUP',
  CoachingEventType.preShotCue => 'PRE_SHOT_CUE',
  CoachingEventType.postShotFeedback => 'POST_SHOT_FEEDBACK',
  CoachingEventType.progression => 'PROGRESSION',
  CoachingEventType.regression => 'REGRESSION',
  CoachingEventType.abstention => 'ABSTENTION',
  CoachingEventType.restSuggestion => 'REST_SUGGESTION',
  CoachingEventType.sessionSummary => 'SESSION_SUMMARY',
  CoachingEventType.systemStatus => 'SYSTEM_STATUS',
};

String _outcomeWire(CoachingOutcomeStatus value) => switch (value) {
  CoachingOutcomeStatus.informational => 'INFORMATIONAL',
  CoachingOutcomeStatus.success => 'SUCCESS',
  CoachingOutcomeStatus.failure => 'FAILURE',
  CoachingOutcomeStatus.abstained => 'ABSTAINED',
  CoachingOutcomeStatus.unsupported => 'UNSUPPORTED',
};

String _deliveryWire(CoachingDeliveryStatus value) => switch (value) {
  CoachingDeliveryStatus.normal => 'NORMAL',
  CoachingDeliveryStatus.degraded => 'DEGRADED',
  CoachingDeliveryStatus.error => 'ERROR',
};

String _roleWire(OverlaySemanticRole value) => switch (value) {
  OverlaySemanticRole.actualCueBallEnd => 'ACTUAL_CUE_BALL_END',
  OverlaySemanticRole.status => 'STATUS',
  OverlaySemanticRole.targetZone => 'TARGET_ZONE',
};

bool _stringListsEqual(List<String> first, List<String> second) {
  if (first.length != second.length) return false;
  for (var index = 0; index < first.length; index += 1) {
    if (first[index] != second[index]) return false;
  }
  return true;
}

Never _invalidValue(String message) =>
    throw ContractFailure(ContractFailureCode.invalidValue, message);

Never _overlayFailure(String message) =>
    throw ContractFailure(ContractFailureCode.overlayGeometry, message);

Never _statusFailure(String message) =>
    throw ContractFailure(ContractFailureCode.statusMismatch, message);

Never _hashFailure(String message) =>
    throw ContractFailure(ContractFailureCode.hashMismatch, message);
