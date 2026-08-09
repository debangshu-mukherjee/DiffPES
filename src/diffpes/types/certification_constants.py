"""Define static identifiers and schema constants for forward certification.

Extended Summary
----------------
This module keeps the declarative vocabulary that the JAX-native certification
and certificate-persistence layers share in the types-owned constant surface.

Routine Listings
----------------
:obj:`CANONICAL_ARRAY_CHUNK_BYTES`
    Array chunk size used by canonical PyTree encoding in bytes.
:obj:`CANONICAL_JSON_PREFIX`
    Domain prefix for canonical JSON identities.
:obj:`CANONICAL_JSON_VERSION`
    Version of the canonical JSON representation.
:obj:`CANONICAL_PYTREE_PREFIX`
    Domain prefix for canonical PyTree identities.
:obj:`CANONICAL_PYTREE_VERSION`
    Version of the canonical PyTree representation.
:obj:`CANONICAL_SUPPORTED_ARRAY_KINDS`
    NumPy dtype kinds accepted by canonical array encoding.
:obj:`CERTIFICATE_ARRAY_KINDS`
    NumPy dtype kinds accepted in persisted certificates.
:obj:`CERTIFICATE_ARRAY_PREVIEW_ITEMS`
    Maximum array elements shown by certificate inspection.
:obj:`CERTIFICATE_DOCUMENT_KEYS`
    Required top-level keys in a certificate document.
:obj:`CERTIFICATE_FORMAT`
    Stable identifier for the forward-certificate document format.
:obj:`CERTIFICATE_H5_GROUP`
    Reserved HDF5 group containing attached certificates.
:obj:`CERTIFICATE_SCHEMA_MAJOR`
    Supported major version of the certificate schema.
:obj:`CERTIFICATE_SCHEMA_MINOR`
    Supported minor version of the certificate schema.
:obj:`CERTIFICATE_SCHEMA_PATTERN`
    Pattern matching supported certificate schema versions.
:obj:`CERTIFICATION_IDENTIFIER_PATTERN`
    Pattern matching permanent certification identifiers.
:obj:`CERTIFICATION_LEVEL_IDS`
    Ordered cumulative scientific-certification level identifiers.
:obj:`CERTIFICATION_LEVEL_PREFIXES`
    Evidence prefixes required by each certification level.
:obj:`CERTIFICATION_LINEAGE_RELATIONSHIPS`
    Typed lineage relationships accepted by certification policy.
:obj:`CERTIFICATION_INDEPENDENT_CLAIM_PREFIXES`
    Claim prefixes requiring independent evidence.
:obj:`CERTIFICATION_POLICY_IDS`
    Stable identifiers of built-in cumulative policies.
:obj:`CERTIFICATION_POLICY_LEVEL_COUNT`
    Number of required levels for each built-in policy.
:obj:`CERTIFICATION_SHARED_RELATIONSHIPS`
    Lineage relationships that share implementation ancestry.
:obj:`CERTIFICATION_SEMVER_PATTERN`
    Pattern matching certification semantic versions.
:obj:`CHECKSUM_ALGORITHM`
    Name of the scientific-identity digest algorithm.
:obj:`CHECKSUM_FILE_CHUNK_BYTES`
    File chunk size used by streaming scientific identities in bytes.
:obj:`CHECKSUM_FORMAT_VERSION`
    Version of the scientific-identity text format.
:obj:`CHECKSUM_PATTERN`
    Pattern matching formatted scientific-identity records.
:obj:`CHECKSUM_RECORD_KIND_PATTERN`
    Pattern matching scientific-identity record-kind identifiers.
:obj:`TB_RADIAL_INPUT_COUNT`
    Number of positional inputs accepted by the radial ARPES model adapter.
:obj:`TB_RADIAL_MODEL_ID`
    Permanent identifier of the radial ARPES forward model.
:obj:`TB_RADIAL_MODEL_VERSION`
    Semantic version of the radial ARPES forward model.

Notes
-----
SHA-256 provides collision-resistant content addressing, not authentication,
authorship, physical validity, or scientific evidence.
"""

import re
from types import MappingProxyType

from beartype.typing import Final, Tuple

CANONICAL_JSON_VERSION: Final[str] = "1"
CANONICAL_PYTREE_VERSION: Final[str] = "1"
CANONICAL_JSON_PREFIX: Final[bytes] = b"DIFFPES-CANONICAL-JSON-V1\x00"
CANONICAL_PYTREE_PREFIX: Final[bytes] = b"DIFFPES-CANONICAL-PYTREE-V1\x00"
CANONICAL_ARRAY_CHUNK_BYTES: Final[int] = 1024 * 1024
CANONICAL_SUPPORTED_ARRAY_KINDS: Final[frozenset[str]] = frozenset(
    {"b", "i", "u", "f", "c"}
)

CHECKSUM_ALGORITHM: Final[str] = "sha256"
CHECKSUM_FORMAT_VERSION: Final[str] = "1"
CHECKSUM_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^sha256:(?P<canonical>[0-9]+):"
    r"(?P<kind>[a-z][a-z0-9-]*):(?P<value>[0-9a-f]{64})$"
)
CHECKSUM_RECORD_KIND_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z0-9-]*$"
)
CHECKSUM_FILE_CHUNK_BYTES: Final[int] = 1024 * 1024

CERTIFICATION_IDENTIFIER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^[a-z][a-z0-9]*(?:[._-][a-z0-9]+)+$"
)
CERTIFICATION_SEMVER_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$"
)

TB_RADIAL_MODEL_ID: Final[str] = "org.diffpes.model.arpes.tb_radial"
TB_RADIAL_MODEL_VERSION: Final[str] = "0.1.0"
TB_RADIAL_INPUT_COUNT: Final[int] = 8

CERTIFICATION_LEVEL_IDS: Final[Tuple[str, ...]] = (
    "identified",
    "validated",
    "differentiable",
    "verified",
    "benchmarked",
    "reproducible",
)
CERTIFICATION_SHARED_RELATIONSHIPS: Final[frozenset[str]] = frozenset(
    {
        "copied-from",
        "derived-from",
        "imports-fixtures-from",
        "shares-generator",
        "shares-implementation",
        "wraps",
    }
)
CERTIFICATION_LINEAGE_RELATIONSHIPS: Final[frozenset[str]] = (
    CERTIFICATION_SHARED_RELATIONSHIPS
    | {
        "independent-derivation",
        "resolves-conflict",
        "resolves-node",
    }
)
CERTIFICATION_INDEPENDENT_CLAIM_PREFIXES: Final[Tuple[str, ...]] = (
    "benchmark",
    "parity",
)
CERTIFICATION_POLICY_IDS: Final[Tuple[str, ...]] = (
    "org.diffpes.policy.exploratory.v1",
    "org.diffpes.policy.research.v1",
    "org.diffpes.policy.publication.v1",
    "org.diffpes.policy.parity.v1",
)
CERTIFICATION_LEVEL_PREFIXES: Final[Tuple[Tuple[str, ...], ...]] = (
    ("identity", "semantic"),
    ("validation", "domain", "output"),
    ("derivative", "differentiability"),
    ("verification", "reference"),
    ("benchmark", "parity"),
    ("reproduction", "environment"),
)
CERTIFICATION_POLICY_LEVEL_COUNT: Final[MappingProxyType] = MappingProxyType(
    {
        "org.diffpes.policy.exploratory.v1": 2,
        "org.diffpes.policy.research.v1": 4,
        "org.diffpes.policy.publication.v1": 6,
        "org.diffpes.policy.parity.v1": 6,
    }
)

CERTIFICATE_FORMAT: Final[str] = "org.diffpes.forward-certificate"
CERTIFICATE_SCHEMA_MAJOR: Final[int] = 1
CERTIFICATE_SCHEMA_MINOR: Final[int] = 0
CERTIFICATE_H5_GROUP: Final[str] = "_diffpes_certificates"
CERTIFICATE_ARRAY_KINDS: Final[frozenset[str]] = frozenset(
    {"b", "i", "u", "f", "c"}
)
CERTIFICATE_DOCUMENT_KEYS: Final[frozenset[str]] = frozenset(
    {
        "format",
        "schema_version",
        "certificate",
        "extensions",
        "consistency_checksum",
    }
)
CERTIFICATE_SCHEMA_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^(?P<major>0|[1-9][0-9]*)"
    r"(?:\.(?P<minor>0|[1-9][0-9]*))?"
    r"(?:\.(?:0|[1-9][0-9]*))?$"
)

CERTIFICATE_ARRAY_PREVIEW_ITEMS: Final[int] = 8


__all__: list[str] = [
    "CANONICAL_ARRAY_CHUNK_BYTES",
    "CANONICAL_JSON_PREFIX",
    "CANONICAL_JSON_VERSION",
    "CANONICAL_PYTREE_PREFIX",
    "CANONICAL_PYTREE_VERSION",
    "CANONICAL_SUPPORTED_ARRAY_KINDS",
    "CERTIFICATE_ARRAY_KINDS",
    "CERTIFICATE_ARRAY_PREVIEW_ITEMS",
    "CERTIFICATE_DOCUMENT_KEYS",
    "CERTIFICATE_FORMAT",
    "CERTIFICATE_H5_GROUP",
    "CERTIFICATE_SCHEMA_MAJOR",
    "CERTIFICATE_SCHEMA_MINOR",
    "CERTIFICATE_SCHEMA_PATTERN",
    "CERTIFICATION_IDENTIFIER_PATTERN",
    "CERTIFICATION_LEVEL_IDS",
    "CERTIFICATION_INDEPENDENT_CLAIM_PREFIXES",
    "CERTIFICATION_LINEAGE_RELATIONSHIPS",
    "CERTIFICATION_LEVEL_PREFIXES",
    "CERTIFICATION_POLICY_IDS",
    "CERTIFICATION_POLICY_LEVEL_COUNT",
    "CERTIFICATION_SHARED_RELATIONSHIPS",
    "CERTIFICATION_SEMVER_PATTERN",
    "CHECKSUM_ALGORITHM",
    "CHECKSUM_FILE_CHUNK_BYTES",
    "CHECKSUM_FORMAT_VERSION",
    "CHECKSUM_PATTERN",
    "CHECKSUM_RECORD_KIND_PATTERN",
    "TB_RADIAL_INPUT_COUNT",
    "TB_RADIAL_MODEL_ID",
    "TB_RADIAL_MODEL_VERSION",
]
