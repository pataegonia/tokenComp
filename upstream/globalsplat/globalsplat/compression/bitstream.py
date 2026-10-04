"""Authenticated scene and residual bitstream containers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class SceneBitstream:
    """The 80-byte ``E2EM0301`` scene container found in the backup."""

    points: int
    channels: int
    rank: int
    mean_fp16: bytes
    score: bytes
    residual: bytes
    flags: int = 0
    version: int = 1

    MAGIC: ClassVar[bytes] = b"E2EM0301"
    HEADER: ClassVar[struct.Struct] = struct.Struct(">8sHHIIIQQQ32s")

    @property
    def bytes_by_stream(self) -> dict[str, int]:
        """Actual entropy strings and overhead, without double-counting containers."""
        y_bytes = z_bytes = 0
        if self.residual:
            residual = ResidualBitstream.unpack(self.residual)
            y_bytes = sum(map(len, residual.y_strings))
            z_bytes = sum(map(len, residual.z_strings))
        return {
            "score": len(self.score),
            "residual_y": y_bytes,
            "residual_z": z_bytes,
            "mean": len(self.mean_fp16),
            "container": self.HEADER.size + len(self.residual) - y_bytes - z_bytes,
        }

    def pack(self) -> bytes:
        metadata = self.HEADER.pack(
            self.MAGIC,
            self.version,
            self.flags,
            self.points,
            self.channels,
            self.rank,
            len(self.mean_fp16),
            len(self.score),
            len(self.residual),
            bytes(32),
        )
        zeroed = metadata + self.mean_fp16 + self.score + self.residual
        digest = hashlib.sha256(zeroed).digest()
        header = metadata[:-32] + digest
        return header + self.mean_fp16 + self.score + self.residual

    @classmethod
    def unpack(cls, data: bytes, *, verify: bool = True) -> "SceneBitstream":
        if len(data) < cls.HEADER.size:
            raise ValueError("scene bitstream is shorter than its 80-byte header")
        (
            magic,
            version,
            flags,
            points,
            channels,
            rank,
            mean_len,
            score_len,
            residual_len,
            digest,
        ) = cls.HEADER.unpack_from(data)
        if magic != cls.MAGIC:
            raise ValueError(f"invalid scene magic {magic!r}")
        expected_size = cls.HEADER.size + mean_len + score_len + residual_len
        if len(data) != expected_size:
            raise ValueError(
                f"scene length mismatch: header declares {expected_size}, got {len(data)}"
            )
        if verify:
            zeroed = data[: cls.HEADER.size - 32] + bytes(32) + data[cls.HEADER.size :]
            actual = hashlib.sha256(zeroed).digest()
            if actual != digest:
                raise ValueError("scene SHA-256 authentication failed")
        cursor = cls.HEADER.size
        mean = data[cursor : cursor + mean_len]
        cursor += mean_len
        score = data[cursor : cursor + score_len]
        cursor += score_len
        residual = data[cursor : cursor + residual_len]
        return cls(points, channels, rank, mean, score, residual, flags, version)


@dataclass(frozen=True, slots=True)
class ResidualBitstream:
    """Portable container for newly encoded hyperprior z/y strings."""

    z_shape: tuple[int, int]
    y_shape: tuple[int, int]
    z_strings: tuple[bytes, ...]
    y_strings: tuple[bytes, ...]

    MAGIC: ClassVar[bytes] = b"M3HPRN01"
    PREFIX: ClassVar[struct.Struct] = struct.Struct(">8sHHIIII")

    @staticmethod
    def _pack_strings(strings: tuple[bytes, ...]) -> bytes:
        return b"".join(struct.pack(">Q", len(value)) + value for value in strings)

    @staticmethod
    def _unpack_strings(data: bytes, cursor: int, count: int) -> tuple[tuple[bytes, ...], int]:
        values: list[bytes] = []
        for _ in range(count):
            if cursor + 8 > len(data):
                raise ValueError("truncated residual string length")
            length = struct.unpack_from(">Q", data, cursor)[0]
            cursor += 8
            end = cursor + length
            if end > len(data):
                raise ValueError("truncated residual entropy string")
            values.append(data[cursor:end])
            cursor = end
        return tuple(values), cursor

    def pack(self) -> bytes:
        prefix = self.PREFIX.pack(
            self.MAGIC,
            1,
            0,
            self.z_shape[0],
            self.z_shape[1],
            self.y_shape[0],
            self.y_shape[1],
        )
        counts = struct.pack(">II", len(self.z_strings), len(self.y_strings))
        return (
            prefix
            + counts
            + self._pack_strings(self.z_strings)
            + self._pack_strings(self.y_strings)
        )

    @classmethod
    def unpack(cls, data: bytes) -> "ResidualBitstream":
        if data.startswith(b"M3HPRANS"):
            raise ValueError(
                "archived M3HPRANS payload detected; its original serializer was not "
                "included in the backup"
            )
        if len(data) < cls.PREFIX.size + 8:
            raise ValueError("residual bitstream is truncated")
        magic, version, flags, zh, zw, yh, yw = cls.PREFIX.unpack_from(data)
        if magic != cls.MAGIC or version != 1 or flags != 0:
            raise ValueError("unsupported residual bitstream")
        z_count, y_count = struct.unpack_from(">II", data, cls.PREFIX.size)
        cursor = cls.PREFIX.size + 8
        z_strings, cursor = cls._unpack_strings(data, cursor, z_count)
        y_strings, cursor = cls._unpack_strings(data, cursor, y_count)
        if cursor != len(data):
            raise ValueError("unexpected trailing bytes in residual bitstream")
        return cls((zh, zw), (yh, yw), z_strings, y_strings)


@dataclass(frozen=True, slots=True)
class Hyper1DSceneBitstream:
    """One-scene E2EH0401 container; only the Morton flag is supported."""

    points: int
    channels: int
    payload: bytes
    flags: int = 0
    version: int = 1

    MAGIC: ClassVar[bytes] = b"E2EH0401"
    HEADER: ClassVar[struct.Struct] = struct.Struct(">8sHHIIQ32s")

    def _validate(self) -> None:
        if self.version != 1 or self.flags & ~1:
            raise ValueError("unsupported Hyper1D version or flags")
        if self.points <= 0 or self.channels <= 0:
            raise ValueError("Hyper1D points/channels must be positive")
        packed = ResidualBitstream.unpack(self.payload)
        if len(packed.z_strings) != 1 or len(packed.y_strings) != 1:
            raise ValueError("Hyper1D container requires one scene (one z/y string)")
        if packed.y_shape[0] != 1 or packed.z_shape[0] != 1 or min(
            packed.y_shape[1], packed.z_shape[1]
        ) <= 0:
            raise ValueError("invalid Hyper1D latent shapes")

    def pack(self) -> bytes:
        self._validate()
        header = self.HEADER.pack(self.MAGIC, self.version, self.flags, self.points,
                                  self.channels, len(self.payload), bytes(32))
        digest = hashlib.sha256(header + self.payload).digest()
        return header[:-32] + digest + self.payload

    @classmethod
    def unpack(cls, data: bytes) -> "Hyper1DSceneBitstream":
        if len(data) < cls.HEADER.size:
            raise ValueError("truncated Hyper1D header")
        magic, version, flags, points, channels, length, digest = cls.HEADER.unpack_from(data)
        if magic != cls.MAGIC:
            raise ValueError(f"invalid Hyper1D magic {magic!r}")
        if len(data) != cls.HEADER.size + length:
            raise ValueError("Hyper1D payload length mismatch")
        zeroed = data[:cls.HEADER.size - 32] + bytes(32) + data[cls.HEADER.size:]
        if hashlib.sha256(zeroed).digest() != digest:
            raise ValueError("Hyper1D SHA-256 authentication failed")
        result = cls(points, channels, data[cls.HEADER.size:], flags, version)
        result._validate()
        return result

    @property
    def bytes_by_stream(self) -> dict[str, int]:
        packed = ResidualBitstream.unpack(self.payload)
        y, z = sum(map(len, packed.y_strings)), sum(map(len, packed.z_strings))
        return {"y": y, "z": z, "container": self.HEADER.size + len(self.payload) - y - z}


@dataclass(frozen=True, slots=True)
class DualHyper1DSceneBitstream:
    """One scene with independently coded base and residual MSH y/z streams."""

    points: int
    channels: int
    base_payload: bytes
    residual_payload: bytes
    flags: int = 0
    version: int = 1

    MAGIC: ClassVar[bytes] = b"E2EH0402"
    HEADER: ClassVar[struct.Struct] = struct.Struct(">8sHHIIQQ32s")

    def _validate(self) -> None:
        for payload in (self.base_payload, self.residual_payload):
            Hyper1DSceneBitstream(self.points, self.channels, payload,
                                 self.flags, self.version)._validate()

    def pack(self) -> bytes:
        self._validate()
        header = self.HEADER.pack(self.MAGIC, self.version, self.flags, self.points,
            self.channels, len(self.base_payload), len(self.residual_payload), bytes(32))
        payload = self.base_payload + self.residual_payload
        digest = hashlib.sha256(header + payload).digest()
        return header[:-32] + digest + payload

    @classmethod
    def unpack(cls, data: bytes) -> "DualHyper1DSceneBitstream":
        if len(data) < cls.HEADER.size:
            raise ValueError("truncated dual Hyper1D header")
        magic, version, flags, points, channels, base_length, residual_length, digest = cls.HEADER.unpack_from(data)
        if magic != cls.MAGIC:
            raise ValueError(f"invalid dual Hyper1D magic {magic!r}")
        if len(data) != cls.HEADER.size + base_length + residual_length:
            raise ValueError("dual Hyper1D payload length mismatch")
        zeroed = data[:cls.HEADER.size - 32] + bytes(32) + data[cls.HEADER.size:]
        if hashlib.sha256(zeroed).digest() != digest:
            raise ValueError("dual Hyper1D SHA-256 authentication failed")
        split = cls.HEADER.size + base_length
        result = cls(points, channels, data[cls.HEADER.size:split], data[split:], flags, version)
        result._validate()
        return result

    @property
    def bytes_by_stream(self) -> dict[str, int]:
        sizes = {}
        for name, payload in (("base", self.base_payload), ("residual", self.residual_payload)):
            packed = ResidualBitstream.unpack(payload)
            sizes[f"{name}_y"] = sum(map(len, packed.y_strings))
            sizes[f"{name}_z"] = sum(map(len, packed.z_strings))
        sizes["container"] = self.HEADER.size + len(self.base_payload) + len(self.residual_payload) - sum(sizes.values())
        return sizes


def scene_bytes_by_stream(data: bytes) -> dict[str, int]:
    if data.startswith(DualHyper1DSceneBitstream.MAGIC):
        return DualHyper1DSceneBitstream.unpack(data).bytes_by_stream
    if data.startswith(Hyper1DSceneBitstream.MAGIC):
        return Hyper1DSceneBitstream.unpack(data).bytes_by_stream
    if data.startswith(SceneBitstream.MAGIC):
        return SceneBitstream.unpack(data).bytes_by_stream
    raise ValueError("unsupported scene bitstream magic")


@dataclass(frozen=True, slots=True)
class ScoreContextBitstream:
    """Container for decoder-causal score-prior entropy strings."""

    rank: int
    slice_channels: int
    flags: int
    strings: tuple[bytes, ...]

    MAGIC: ClassVar[bytes] = b"SCCTX001"
    PREFIX: ClassVar[struct.Struct] = struct.Struct(">8sHHIII")

    def pack(self) -> bytes:
        prefix = self.PREFIX.pack(
            self.MAGIC,
            1,
            self.flags,
            self.rank,
            self.slice_channels,
            len(self.strings),
        )
        return prefix + b"".join(
            struct.pack(">Q", len(value)) + value for value in self.strings
        )

    @classmethod
    def unpack(cls, data: bytes) -> "ScoreContextBitstream":
        if len(data) < cls.PREFIX.size:
            raise ValueError("contextual score bitstream is truncated")
        magic, version, flags, rank, slice_channels, count = cls.PREFIX.unpack_from(data)
        if magic != cls.MAGIC or version != 1:
            raise ValueError("unsupported contextual score bitstream")
        strings, cursor = ResidualBitstream._unpack_strings(data, cls.PREFIX.size, count)
        if cursor != len(data):
            raise ValueError("unexpected trailing bytes in contextual score bitstream")
        return cls(rank, slice_channels, flags, strings)
