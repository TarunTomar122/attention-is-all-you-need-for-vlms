"""Extract a small image subset from a streaming ZIP when no central directory is available."""

from __future__ import annotations

import argparse
import struct
import zlib
from pathlib import Path


LOCAL = b"PK" + bytes.fromhex("0304")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--archive", type=Path, required=True); parser.add_argument("--list", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    wanted = {f"images/{name}" for name in args.list.read_text().splitlines() if name}
    args.output.mkdir(parents=True, exist_ok=True); found = set(); count = 0
    with args.archive.open("rb", buffering=1024 * 1024) as source:
        while True:
            signature = source.read(4)
            if signature != LOCAL: break
            header = source.read(26)
            if len(header) != 26: raise ValueError("truncated local ZIP header")
            _, flags, method, _, _, _, compressed, uncompressed, name_len, extra_len = struct.unpack("<5H3L2H", header)
            name = source.read(name_len).decode("utf-8")
            source.seek(extra_len, 1)
            data = source.read(compressed)
            if len(data) != compressed: raise ValueError(f"truncated ZIP entry {name}")
            if name not in wanted: continue
            if flags & 0x08: raise ValueError(f"data-descriptor entry unsupported: {name}")
            payload = data if method == 0 else zlib.decompress(data, -15) if method == 8 else None
            if payload is None or len(payload) != uncompressed: raise ValueError(f"unsupported or corrupt ZIP entry {name}")
            (args.output / Path(name).name).write_bytes(payload); found.add(name); count += 1
    missing = wanted - found
    if missing: raise FileNotFoundError(f"{len(missing)} requested GQA images missing; first={sorted(missing)[:3]}")
    print(f"extracted {count} images")


if __name__ == "__main__": main()
