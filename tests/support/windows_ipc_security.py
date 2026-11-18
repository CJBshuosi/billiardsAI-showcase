"""Read-only, test-only validation of CPython 3.12.10's private IPC ACLs.

Trust SYSTEM and Builtin Administrators, but no other ordinary user. Nothing is
loaded from Win32 at import time. Synthetic predicate/API tests are not native
descriptor evidence. See CPython v3.12.10 Modules/posixmodule.c os_mkdir_impl and
Microsoft's GetSecurityInfo, GetTokenInformation and CreateFileW contracts.
"""

from __future__ import annotations

import ctypes as C
import os
import stat
import sys
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

SYSTEM = "S-1-5-18"
ADMIN = "S-1-5-32-544"
OWNER_RIGHTS = "S-1-3-4"


@dataclass(frozen=True)
class TokenIdentity:
    user: str
    owner: str


@dataclass(frozen=True)
class Ace:
    type: int
    flags: int
    mask: int
    sid: str


@dataclass(frozen=True)
class Descriptor:
    owner: str
    control: int
    dacl_present: bool
    dacl_null: bool
    aces: tuple[Ace, ...]


def evaluate_descriptor(value, token, *, directory):
    """Fail closed on any semantics outside the reviewed private ACL shape."""
    assert value.owner == token.user or (value.owner == ADMIN and token.owner == ADMIN), (
        "object owner does not match current token user/default administrator owner"
    )
    assert value.dacl_present and value.control & 4, "missing DACL"
    assert not value.dacl_null and value.aces, "null or empty DACL"
    if directory:
        assert value.control & 0x1000, "directory DACL is not protected"
    approved = {value.owner, OWNER_RIGHTS, SYSTEM, ADMIN}
    owner_access = False
    for ace in value.aces:
        assert ace.type == 0, "unsupported ACE type (only basic allow ACEs are accepted)"
        assert ace.sid in approved, "unapproved SID grant"
        assert ace.flags == 3 if directory else ace.flags in (0, 16), (
            "unsupported ACE inheritance or flags"
        )
        assert ace.mask == 0x1F01FF, "grant lacks exact file full-control semantics"
        if ace.sid in (value.owner, OWNER_RIGHTS):
            owner_access = True
    assert owner_access, "no usable full owner access"


# Windows LLP64 uses 32-bit DWORD/LONG even on AMD64 (including when tested on
# a platform where ctypes.c_long is 64-bit). Handles and pointers remain native.
_DWORD = C.c_uint32
_WORD = C.c_uint16
_BYTE = C.c_uint8
_BOOL = C.c_int32
_PTR = C.c_void_p
_PPTR = C.POINTER(_PTR)


class _FileInformation(C.Structure):
    _fields_ = [
        (name, _DWORD)
        for name in (
            "attributes",
            "creation_low",
            "creation_high",
            "access_low",
            "access_high",
            "write_low",
            "write_high",
            "volume",
            "size_high",
            "size_low",
            "links",
            "index_high",
            "index_low",
        )
    ]


class _Acl(C.Structure):
    _fields_ = [
        ("revision", _BYTE),
        ("reserved", _BYTE),
        ("size", _WORD),
        ("count", _WORD),
        ("reserved2", _WORD),
    ]


class _AceHeader(C.Structure):
    _fields_ = [("type", _BYTE), ("flags", _BYTE), ("size", _WORD)]


def _check(result, operation):
    if not result:
        raise OSError(f"{operation} failed")


class _Win32:
    """Own every opened handle and LocalAlloc buffer; never read IPC contents."""

    def __init__(self):
        # Restricted to system DLLs; constructor is reached only on pinned Win32.
        self.k = C.WinDLL("kernel32", use_last_error=True)
        self.a = C.WinDLL("advapi32", use_last_error=True)
        signatures = [
            (self.k, "GetCurrentProcess", _PTR, []),
            (self.k, "GetCurrentThread", _PTR, []),
            (self.k, "CloseHandle", _BOOL, [_PTR]),
            (self.k, "LocalFree", _PTR, [_PTR]),
            (
                self.k,
                "CreateFileW",
                _PTR,
                [C.c_wchar_p, _DWORD, _DWORD, _PTR, _DWORD, _DWORD, _PTR],
            ),
            (self.k, "GetFileInformationByHandle", _BOOL, [_PTR, C.POINTER(_FileInformation)]),
            (self.a, "OpenProcessToken", _BOOL, [_PTR, _DWORD, _PPTR]),
            (self.a, "OpenThreadToken", _BOOL, [_PTR, _DWORD, _BOOL, _PPTR]),
            (self.a, "GetTokenInformation", _BOOL, [_PTR, _DWORD, _PTR, _DWORD, C.POINTER(_DWORD)]),
            (
                self.a,
                "GetSecurityInfo",
                _DWORD,
                [_PTR, _DWORD, _DWORD, _PPTR, _PPTR, _PPTR, _PPTR, _PPTR],
            ),
            (self.a, "IsValidSecurityDescriptor", _BOOL, [_PTR]),
            (self.a, "GetSecurityDescriptorLength", _DWORD, [_PTR]),
            (
                self.a,
                "GetSecurityDescriptorControl",
                _BOOL,
                [_PTR, C.POINTER(_WORD), C.POINTER(_DWORD)],
            ),
            (
                self.a,
                "GetSecurityDescriptorDacl",
                _BOOL,
                [_PTR, C.POINTER(_BOOL), _PPTR, C.POINTER(_BOOL)],
            ),
            (self.a, "IsValidAcl", _BOOL, [_PTR]),
            (self.a, "GetAce", _BOOL, [_PTR, _DWORD, _PPTR]),
            (self.a, "IsValidSid", _BOOL, [_PTR]),
            (self.a, "GetLengthSid", _DWORD, [_PTR]),
            (self.a, "ConvertSidToStringSidW", _BOOL, [_PTR, _PPTR]),
        ]
        for dll, name, result, arguments in signatures:
            function = getattr(dll, name)
            function.restype = result
            function.argtypes = arguments

    def close(self, handle):
        _check(self.k.CloseHandle(handle), "CloseHandle")

    def _free(self, pointer):
        if pointer.value and self.k.LocalFree(pointer):
            raise OSError("LocalFree failed")

    @staticmethod
    def _copy_string(pointer):
        return C.wstring_at(pointer)

    def _sid_string(self, sid):
        text = _PTR()
        try:
            _check(self.a.ConvertSidToStringSidW(sid, C.byref(text)), "ConvertSidToStringSidW")
            assert text.value, "missing SID string"
            return self._copy_string(text)
        finally:
            self._free(text)

    def _bounded_sid(self, pointer, begin, end):
        assert pointer and begin <= pointer <= end - 8, "SID outside returned buffer"
        # Check SID's variable-length subauthority region before native SID APIs.
        count = _BYTE.from_address(pointer + 1).value
        assert count <= 15 and pointer + 8 + 4 * count <= end, "truncated SID"
        _check(self.a.IsValidSid(pointer), "IsValidSid")
        assert self.a.GetLengthSid(pointer) == 8 + 4 * count, "invalid SID size"
        return self._sid_string(pointer)

    def _open_token(self):
        token = _PTR()
        if self.a.OpenThreadToken(self.k.GetCurrentThread(), 8, True, C.byref(token)):
            # This contract is for the creating process; an impersonating thread
            # needs a separately reviewed creation/owner model, so fail closed.
            self.close(token)
            raise AssertionError("impersonating thread is outside the IPC test contract")
        if C.get_last_error() != 1008:  # ERROR_NO_TOKEN; do not hide access denial.
            raise OSError("OpenThreadToken failed")
        _check(
            self.a.OpenProcessToken(self.k.GetCurrentProcess(), 8, C.byref(token)),
            "OpenProcessToken",
        )
        assert token.value, "missing process token"
        return token.value

    def _token_sid(self, token, information_class):
        length = _DWORD()
        ok = self.a.GetTokenInformation(token, information_class, None, 0, C.byref(length))
        assert not ok and C.get_last_error() == 122, "unexpected token sizing result"
        assert C.sizeof(_PTR) <= length.value <= 65536, "invalid token buffer size"
        buffer = C.create_string_buffer(length.value)
        capacity = length.value
        _check(
            self.a.GetTokenInformation(token, information_class, buffer, capacity, C.byref(length)),
            "GetTokenInformation",
        )
        assert C.sizeof(_PTR) <= length.value <= capacity, "invalid token result length"
        begin = C.addressof(buffer)
        sid = _PTR.from_buffer(buffer).value  # first field in TOKEN_USER and TOKEN_OWNER
        return self._bounded_sid(sid, begin, begin + length.value)

    def token_identity(self):
        token = self._open_token()
        try:
            return TokenIdentity(self._token_sid(token, 1), self._token_sid(token, 4))
        finally:
            self.close(token)

    def open(self, path):
        # READ_CONTROL | FILE_READ_ATTRIBUTES; OPEN_EXISTING; no content access,
        # no following the final reparse point, no FILE_SHARE_DELETE. Holding
        # both handles prevents rename/deletion throughout descriptor checks.
        handle = self.k.CreateFileW(str(path), 0x20080, 3, None, 3, 0x02200000, None)
        if handle is None or handle == _PTR(-1).value:
            raise OSError("CreateFileW read-only metadata open failed")
        return handle

    def identity(self, handle, *, directory):
        info = _FileInformation()
        _check(
            self.k.GetFileInformationByHandle(handle, C.byref(info)), "GetFileInformationByHandle"
        )
        assert not info.attributes & (0x400 | 0x40), "reparse/device object"
        assert bool(info.attributes & 0x10) == directory, "unexpected object type"
        assert directory or info.links == 1, "request has multiple hard links"
        index = (info.index_high << 32) | info.index_low
        assert index, "missing stable file identity"
        return (info.volume, index)

    def descriptor(self, handle):
        owner, dacl, sd = _PTR(), _PTR(), _PTR()
        try:
            status = self.a.GetSecurityInfo(
                handle, 1, 5, C.byref(owner), None, C.byref(dacl), None, C.byref(sd)
            )
            if status:
                raise OSError(f"GetSecurityInfo failed ({status})")
            assert sd.value, "missing security descriptor"
            return self._decode_descriptor(sd, owner, dacl)
        finally:
            # Owner/DACL are borrowed pointers into this one LocalAlloc buffer.
            self._free(sd)

    def _decode_descriptor(self, sd, owner, returned_dacl):
        _check(self.a.IsValidSecurityDescriptor(sd), "IsValidSecurityDescriptor")
        size = self.a.GetSecurityDescriptorLength(sd)
        assert 20 <= size <= 1024 * 1024, "invalid security descriptor size"
        begin, end = sd.value, sd.value + size
        owner_sid = self._bounded_sid(owner.value, begin, end)
        control, revision = _WORD(), _DWORD()
        _check(
            self.a.GetSecurityDescriptorControl(sd, C.byref(control), C.byref(revision)),
            "GetSecurityDescriptorControl",
        )
        assert revision.value == 1, "unsupported descriptor revision"
        present, defaulted, dacl = _BOOL(), _BOOL(), _PTR()
        _check(
            self.a.GetSecurityDescriptorDacl(
                sd, C.byref(present), C.byref(dacl), C.byref(defaulted)
            ),
            "GetSecurityDescriptorDacl",
        )
        aces = []
        if present.value and dacl.value:
            assert dacl.value == returned_dacl.value, "inconsistent DACL pointers"
            assert begin <= dacl.value <= end - C.sizeof(_Acl), "DACL outside descriptor"
            header = _Acl.from_address(dacl.value)
            assert header.revision == 2, "unsupported ACL revision"
            assert C.sizeof(_Acl) <= header.size <= end - dacl.value, "truncated DACL"
            _check(self.a.IsValidAcl(dacl), "IsValidAcl")
            acl_end = dacl.value + header.size
            for index in range(header.count):
                pointer = _PTR()
                _check(self.a.GetAce(dacl, index, C.byref(pointer)), "GetAce")
                address = pointer.value
                assert address and dacl.value + 8 <= address <= acl_end - 4, "invalid ACE pointer"
                ace = _AceHeader.from_address(address)
                assert ace.type == 0, "unsupported native ACE type"
                assert 16 <= ace.size <= acl_end - address, "truncated allow ACE"
                mask = _DWORD.from_address(address + 4).value
                sid = self._bounded_sid(address + 8, address + 8, address + ace.size)
                # Basic ACEs carry exactly the SID; reject appended semantics.
                assert self.a.GetLengthSid(address + 8) + 8 == ace.size, "unexpected ACE payload"
                aces.append(Ace(ace.type, ace.flags, mask, sid))
        return Descriptor(
            owner_sid, control.value, bool(present.value), not dacl.value, tuple(aces)
        )


def _path_identity(path, *, directory):
    info = path.lstat()
    assert stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode), (
        "unexpected path type or symbolic link"
    )
    assert not getattr(info, "st_file_attributes", 0) & 0x400, "reparse path"
    assert directory or info.st_nlink == 1, "request hard link"
    assert info.st_ino, "missing path identity"
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode), info.st_nlink


def _assert_private_pair(root, request, reader):
    assert root.is_absolute() and request == root / "request", "unexpected owned IPC paths"
    paths = ((root, True), (request, False))
    before = [_path_identity(path, directory=directory) for path, directory in paths]
    token = reader.token_identity()  # independent of observed object owners
    result = {"contract": "private-owner-system-admin", "token": asdict(token)}
    with ExitStack() as resources:
        handles = []
        for path, _ in paths:
            handle = reader.open(path)
            resources.callback(reader.close, handle)
            handles.append(handle)
        identities = [
            reader.identity(handle, directory=directory)
            for handle, (_, directory) in zip(handles, paths, strict=True)
        ]
        for name, handle, (path, directory), identity in zip(
            ("root", "request"), handles, paths, identities, strict=True
        ):
            value = reader.descriptor(handle)
            evaluate_descriptor(value, token, directory=directory)
            result[name] = {"path": str(path), "identity": identity, "descriptor": asdict(value)}
        for (path, directory), handle, original, identity in zip(
            paths, handles, before, identities, strict=True
        ):
            assert _path_identity(path, directory=directory) == original, "path identity changed"
            assert reader.identity(handle, directory=directory) == identity, (
                "handle identity changed"
            )
            probe = reader.open(path)
            try:
                assert reader.identity(probe, directory=directory) == identity, (
                    "path identity changed"
                )
            finally:
                reader.close(probe)
    return result


def _supported_runtime():
    return (
        os.name == "nt"
        and sys.implementation.name == "cpython"
        and sys.version_info[:3] == (3, 12, 10)
    )


def assert_private_windows_ipc(root: Path, request: Path) -> dict[str, object]:
    """Validate both owned objects, returning synthetic-only descriptor metadata."""
    assert _supported_runtime(), "Windows IPC ACL evidence requires Windows CPython 3.12.10"
    return _assert_private_pair(root, request, _Win32())
