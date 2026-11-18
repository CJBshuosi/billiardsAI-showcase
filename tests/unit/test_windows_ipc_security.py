"""Synthetic ACL predicates/API seams; these do not exercise Windows security."""

from __future__ import annotations

import ctypes as C
import importlib
import importlib.util
import json
import struct
from dataclasses import replace
from types import SimpleNamespace

import pytest

USER = "S-1-5-21-111-222-333-1001"
OTHER = "S-1-5-21-111-222-333-1002"
ADMIN = "S-1-5-32-544"
SYSTEM = "S-1-5-18"
OWNER_RIGHTS = "S-1-3-4"


@pytest.fixture
def security():
    assert importlib.util.find_spec("tests.support.windows_ipc_security") is not None, (
        "missing Windows IPC security predicate and reader"
    )
    return importlib.import_module("tests.support.windows_ipc_security")


def descriptor(security, *, directory=True, owner=USER):
    return security.Descriptor(
        owner=owner,
        control=0x1004 if directory else 0x0404,
        dacl_present=True,
        dacl_null=False,
        aces=tuple(
            security.Ace(0, 3 if directory else 16, 0x1F01FF, sid)
            for sid in (SYSTEM, ADMIN, OWNER_RIGHTS)
        ),
    )


@pytest.mark.parametrize("directory", [True, False])
@pytest.mark.parametrize("owner,token_owner", [(USER, USER), (ADMIN, ADMIN)])
def test_accepts_private_descriptor_for_independently_validated_owner(
    security, directory, owner, token_owner
):
    security.evaluate_descriptor(
        descriptor(security, directory=directory, owner=owner),
        security.TokenIdentity(USER, token_owner),
        directory=directory,
    )


@pytest.mark.parametrize("sid", ["S-1-1-0", "S-1-5-32-545", "S-1-5-11", OTHER])
@pytest.mark.parametrize("directory", [True, False])
def test_rejects_broad_or_foreign_grant_even_when_owner_access_is_present(security, sid, directory):
    value = descriptor(security, directory=directory)
    value = replace(value, aces=(*value.aces, security.Ace(0, 0, 1, sid)))
    with pytest.raises(AssertionError):
        security.evaluate_descriptor(value, security.TokenIdentity(USER, USER), directory=directory)


@pytest.mark.parametrize(
    "change",
    [
        {"dacl_present": False},
        {"dacl_null": True},
        {"aces": ()},
        {"control": 4},
        {"control": 0x1000},
        {"owner": OTHER},
        {"owner": ADMIN},
    ],
)
def test_rejects_missing_empty_unprotected_or_wrong_owner_directory(security, change):
    with pytest.raises(AssertionError):
        security.evaluate_descriptor(
            replace(descriptor(security), **change),
            security.TokenIdentity(USER, USER),
            directory=True,
        )


def test_foreign_default_token_owner_does_not_whitelist_foreign_object_owner(security):
    with pytest.raises(AssertionError):
        security.evaluate_descriptor(
            descriptor(security, owner=OTHER),
            security.TokenIdentity(USER, OTHER),
            directory=True,
        )


@pytest.mark.parametrize("kind", [1, 5, 9, 255])
def test_rejects_deny_object_callback_and_unknown_ace_semantics(security, kind):
    value = descriptor(security)
    with pytest.raises(AssertionError):
        security.evaluate_descriptor(
            replace(value, aces=(*value.aces, security.Ace(kind, 3, 0x1F01FF, USER))),
            security.TokenIdentity(USER, USER),
            directory=True,
        )


@pytest.mark.parametrize("flags", [0, 1, 2, 7, 11, 19, 0x43])
def test_directory_requires_usable_full_inheritance_without_unknown_flags(security, flags):
    value = descriptor(security)
    with pytest.raises(AssertionError):
        security.evaluate_descriptor(
            replace(value, aces=(*value.aces[:2], replace(value.aces[2], flags=flags))),
            security.TokenIdentity(USER, USER),
            directory=True,
        )


@pytest.mark.parametrize("directory", [True, False])
@pytest.mark.parametrize("mask", [0, 0x120089, 0x1F01FE, 0xFFFFFFFF])
def test_requires_full_owner_control_and_rejects_unknown_access_bits(security, directory, mask):
    value = descriptor(security, directory=directory)
    with pytest.raises(AssertionError):
        security.evaluate_descriptor(
            replace(value, aces=(*value.aces[:2], replace(value.aces[2], mask=mask))),
            security.TokenIdentity(USER, USER),
            directory=directory,
        )


@pytest.mark.parametrize("flags", [3, 8, 24, 0x40])
def test_request_rejects_inherit_only_and_unsupported_flags(security, flags):
    value = descriptor(security, directory=False)
    with pytest.raises(AssertionError):
        security.evaluate_descriptor(
            replace(value, aces=(*value.aces[:2], replace(value.aces[2], flags=flags))),
            security.TokenIdentity(USER, USER),
            directory=False,
        )


def test_owner_access_cannot_be_inferred_from_system_and_admin_grants(security):
    value = descriptor(security)
    with pytest.raises(AssertionError):
        security.evaluate_descriptor(
            replace(value, aces=value.aces[:2]), security.TokenIdentity(USER, USER), directory=True
        )


class SyntheticReader:
    """Only the OS boundary is replaced; pair validation and cleanup stay real."""

    def __init__(self, security, root, request):
        self.security = security
        self.root = root
        self.request = request
        self.closed = []
        self.reads = []
        self.replaced = False
        self.failure = None

    def token_identity(self):
        return self.security.TokenIdentity(USER, USER)

    def open(self, path):
        return path

    def close(self, handle):
        self.closed.append(handle)

    def identity(self, handle, *, directory):
        if handle == self.request and self.replaced:
            return (1, 999)
        return (1, 10 if directory else 11)

    def descriptor(self, handle):
        self.reads.append(handle)
        if self.failure:
            raise self.failure
        return descriptor(self.security, directory=handle == self.root)


def pair(tmp_path):
    root = tmp_path / "owned-ipc"
    root.mkdir()
    request = root / "request"
    request.write_text("synthetic-secret-never-in-evidence", encoding="utf-8")
    return root, request


def test_pair_checks_both_descriptors_and_returns_metadata_only(security, tmp_path):
    root, request = pair(tmp_path)
    reader = SyntheticReader(security, root, request)
    evidence = security._assert_private_pair(root, request, reader)
    encoded = json.dumps(evidence)
    assert evidence["contract"] == "private-owner-system-admin"
    assert evidence["root"]["descriptor"]["owner"] == USER
    assert evidence["request"]["descriptor"]["aces"][2]["sid"] == OWNER_RIGHTS
    assert "synthetic-secret" not in encoded
    assert reader.reads == [root, request]
    assert reader.closed.count(root) == 2 and reader.closed.count(request) == 2


def test_query_failure_propagates_and_closes_all_opened_handles(security, tmp_path):
    root, request = pair(tmp_path)
    reader = SyntheticReader(security, root, request)
    reader.failure = OSError("ACL query failed")
    with pytest.raises(OSError, match="ACL query failed"):
        security._assert_private_pair(root, request, reader)
    assert sorted(reader.closed) == sorted([root, request])


def test_private_directory_does_not_hide_broad_request_acl(security, tmp_path):
    root, request = pair(tmp_path)
    reader = SyntheticReader(security, root, request)
    original = reader.descriptor

    def broad_request(handle):
        value = original(handle)
        if handle == request:
            value = replace(value, aces=(*value.aces, security.Ace(0, 16, 0x1F01FF, OTHER)))
        return value

    reader.descriptor = broad_request
    with pytest.raises(AssertionError, match="unapproved SID"):
        security._assert_private_pair(root, request, reader)
    assert sorted(reader.closed) == sorted([root, request])


def test_rejects_path_replacement_around_descriptor_read(security, tmp_path):
    root, request = pair(tmp_path)
    reader = SyntheticReader(security, root, request)
    original = reader.descriptor

    def replace_after_read(handle):
        value = original(handle)
        reader.replaced = True
        return value

    reader.descriptor = replace_after_read
    with pytest.raises(AssertionError, match="identity"):
        security._assert_private_pair(root, request, reader)
    assert reader.closed.count(root) >= 1 and reader.closed.count(request) >= 1


@pytest.mark.parametrize("replacement", ["symlink", "hardlink", "directory"])
def test_rejects_nonregular_linked_request_before_descriptor_query(security, tmp_path, replacement):
    root, request = pair(tmp_path)
    request.unlink()
    other = tmp_path / "other"
    other.write_text("synthetic", encoding="utf-8")
    if replacement == "symlink":
        request.symlink_to(other)
    elif replacement == "hardlink":
        request.hardlink_to(other)
    else:
        request.mkdir()
    reader = SyntheticReader(security, root, request)
    with pytest.raises(AssertionError):
        security._assert_private_pair(root, request, reader)
    assert not reader.reads


def test_rejects_request_outside_owned_root(security, tmp_path):
    root, request = pair(tmp_path)
    reader = SyntheticReader(security, root, request)
    with pytest.raises(AssertionError):
        security._assert_private_pair(root, tmp_path / "request", reader)
    assert not reader.reads


@pytest.mark.parametrize("fail", [False, True])
def test_native_sid_string_allocation_is_freed_on_success_and_copy_failure(security, fail):
    value = C.create_unicode_buffer(USER)
    freed = []
    reader = object.__new__(security._Win32)

    def convert(sid, output):
        C.cast(output, C.POINTER(C.c_void_p))[0] = C.addressof(value)
        return 1

    reader.a = SimpleNamespace(ConvertSidToStringSidW=convert)
    reader.k = SimpleNamespace(LocalFree=lambda ptr: freed.append(ptr.value) or None)
    reader._copy_string = lambda ptr: (_ for _ in ()).throw(ValueError("copy")) if fail else USER
    if fail:
        with pytest.raises(ValueError, match="copy"):
            reader._sid_string(123)
    else:
        assert reader._sid_string(123) == USER
    assert freed == [C.addressof(value)]


@pytest.mark.parametrize("status", [0, 5])
def test_native_descriptor_buffer_freed_even_when_query_or_decode_fails(security, status):
    freed = []
    reader = object.__new__(security._Win32)

    def get_security(handle, kind, flags, owner, group, dacl, sacl, output):
        C.cast(output, C.POINTER(C.c_void_p))[0] = 1234
        return status

    reader.a = SimpleNamespace(GetSecurityInfo=get_security)
    reader.k = SimpleNamespace(LocalFree=lambda ptr: freed.append(ptr.value) or None)
    reader._decode_descriptor = lambda *args: (_ for _ in ()).throw(ValueError("decode"))
    with pytest.raises((ValueError, OSError)):
        reader.descriptor(123)
    assert freed == [1234]


@pytest.mark.parametrize("fail", [False, True])
def test_native_token_handle_closed_after_both_identity_queries(security, fail):
    closed, queried = [], []
    reader = object.__new__(security._Win32)
    reader._open_token = lambda: 654

    def token_sid(handle, information_class):
        queried.append((handle, information_class))
        if fail:
            raise OSError("token query")
        return USER

    reader._token_sid = token_sid
    reader.k = SimpleNamespace(CloseHandle=lambda handle: closed.append(handle) or 1)
    if fail:
        with pytest.raises(OSError, match="token query"):
            reader.token_identity()
    else:
        assert reader.token_identity() == security.TokenIdentity(USER, USER)
        assert queried == [(654, 1), (654, 4)]
    assert closed == [654]


def test_native_cleanup_failures_are_not_reported_as_success(security):
    reader = object.__new__(security._Win32)
    reader.k = SimpleNamespace(LocalFree=lambda ptr: ptr.value, CloseHandle=lambda handle: 0)
    with pytest.raises(OSError, match="LocalFree"):
        reader._free(C.c_void_p(123))
    with pytest.raises(OSError, match="CloseHandle"):
        reader.close(123)


@pytest.mark.parametrize(
    "attributes,links,directory", [(0x410, 1, True), (0x10, 1, False), (0, 2, False)]
)
def test_native_identity_rejects_reparse_directory_and_multilink_file(
    security, attributes, links, directory
):
    reader = object.__new__(security._Win32)

    def information(handle, output):
        value = C.cast(output, C.POINTER(security._FileInformation)).contents
        value.attributes = attributes
        value.links = links
        value.volume = 1
        value.index_low = 2
        return 1

    reader.k = SimpleNamespace(GetFileInformationByHandle=information)
    with pytest.raises(AssertionError):
        reader.identity(123, directory=directory)


def test_file_identity_uses_full_64_bit_index_without_dword_truncation(security):
    reader = object.__new__(security._Win32)

    def information(handle, output):
        value = C.cast(output, C.POINTER(security._FileInformation)).contents
        value.links = 1
        value.volume = 0x12345678
        value.index_high = 0xABCDEF01
        value.index_low = 0x23456789
        return 1

    reader.k = SimpleNamespace(GetFileInformationByHandle=information)
    assert reader.identity(123, directory=False) == (0x12345678, 0xABCDEF0123456789)


@pytest.mark.parametrize("corruption", [None, "unknown-ace", "short-ace", "long-sid"])
def test_decodes_bounded_native_acl_layout_and_rejects_unsupported_bytes(security, corruption):
    # Hand-built self-relative SD: 20-byte header, 28-byte user SID, then
    # 8-byte ACL header and 20/24/20-byte basic allow ACEs (Windows LLP64).
    user = bytes.fromhex("0105000000000005150000006f000000de0000004d010000e9030000")
    system = bytes.fromhex("010100000000000512000000")
    admin = bytes.fromhex("01020000000000052000000020020000")
    owner_rights = bytes.fromhex("010100000000000304000000")
    aces = b"".join(
        struct.pack("<BBHI", 0, 3, 8 + len(sid), 0x1F01FF) + sid
        for sid in (system, admin, owner_rights)
    )
    raw = bytearray(struct.pack("<BBHLLLL", 1, 0, 0x1004, 20, 0, 0, 48) + user)
    raw.extend(struct.pack("<BBHHH", 2, 0, 72, 3, 0) + aces)
    if corruption == "unknown-ace":
        raw[56] = 9
    elif corruption == "short-ace":
        raw[58:60] = struct.pack("<H", 8)
    elif corruption == "long-sid":
        raw[65] = 15
    buffer = C.create_string_buffer(bytes(raw))
    base = C.addressof(buffer)
    sid_strings = {base + 20: USER, base + 64: SYSTEM, base + 84: ADMIN, base + 108: OWNER_RIGHTS}

    def control(sd, result, revision):
        C.cast(result, C.POINTER(C.c_uint16))[0] = 0x1004
        C.cast(revision, C.POINTER(C.c_uint32))[0] = 1
        return 1

    def dacl(sd, present, result, defaulted):
        C.cast(present, C.POINTER(C.c_int32))[0] = 1
        C.cast(result, C.POINTER(C.c_void_p))[0] = base + 48
        return 1

    def ace(acl, index, result):
        C.cast(result, C.POINTER(C.c_void_p))[0] = base + (56, 76, 100)[index]
        return 1

    reader = object.__new__(security._Win32)
    reader.a = SimpleNamespace(
        IsValidSecurityDescriptor=lambda sd: 1,
        GetSecurityDescriptorLength=lambda sd: 120,
        GetSecurityDescriptorControl=control,
        GetSecurityDescriptorDacl=dacl,
        IsValidAcl=lambda acl: 1,
        GetAce=ace,
        IsValidSid=lambda sid: 1,
        GetLengthSid=lambda sid: 8 + 4 * C.c_uint8.from_address(sid + 1).value,
    )
    reader._sid_string = lambda sid: sid_strings[sid]
    if corruption:
        with pytest.raises(AssertionError):
            reader._decode_descriptor(
                C.c_void_p(base), C.c_void_p(base + 20), C.c_void_p(base + 48)
            )
    else:
        actual = reader._decode_descriptor(
            C.c_void_p(base), C.c_void_p(base + 20), C.c_void_p(base + 48)
        )
        assert actual == descriptor(security)
        security.evaluate_descriptor(actual, security.TokenIdentity(USER, USER), directory=True)


def test_non_windows_entry_refuses_before_loading_native_apis(security, monkeypatch, tmp_path):
    monkeypatch.setattr(security, "_supported_runtime", lambda: False)
    monkeypatch.setattr(security, "_Win32", lambda: pytest.fail("must not load native APIs"))
    with pytest.raises(AssertionError, match="CPython 3.12.10"):
        security.assert_private_windows_ipc(tmp_path, tmp_path / "request")
