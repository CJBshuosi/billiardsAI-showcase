enum ContractFailureCode {
  sizeLimitExceeded,
  invalidUtf8,
  invalidJson,
  duplicateKey,
  invalidNumber,
  nonCanonicalJson,
  unsupportedSchemaVersion,
  unknownField,
  missingField,
  invalidValue,
  invalidUuid,
  invalidDigest,
  statusMismatch,
  lineageMismatch,
  overlayGeometry,
  hashMismatch,
}

final class ContractFailure implements Exception {
  const ContractFailure(this.code, this.message);

  final ContractFailureCode code;
  final String message;

  @override
  String toString() => 'ContractFailure(${code.name}): $message';
}
