"""Stream the GQA ZIP and retain only the FineCops image subset."""

from __future__ import annotations

import argparse
import struct
import urllib.request
import zlib
from pathlib import Path


LOCAL = b"PK" + bytes.fromhex("0304")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--url", required=True); parser.add_argument("--list", type=Path, required=True); parser.add_argument("--output", type=Path, required=True); args = parser.parse_args()
    wanted = {f"images/{name}" for name in args.list.read_text().splitlines() if name}
    args.output.mkdir(parents=True, exist_ok=True); found = set(); entries = 0
    with urllib.request.urlopen(args.url, timeout=120) as source:
        while True:
            signature = source.read(4)
            if not signature: break
            if signature != LOCAL: break
            header = source.read(26)
            if len(header) != 26: raise ValueError("truncated local ZIP header")
            _, flags, method, _, _, _, compressed, uncompressed, name_len, extra_len = struct.unpack("<5H3L2H", header)
            name = source.read(name_len).decode("utf-8"); source.read(extra_len)
            if flags & 0x08: raise ValueError(f"data-descriptor entry unsupported: {name}")
            remaining = compressed; chunks = []
            while remaining:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk: raise ValueError(f"truncated ZIP entry {name}")
                remaining -= len(chunk)
                if name in wanted: chunks.append(chunk)
            if name in wanted:
                data = b"".join(chunks)
                payload = data if method == 0 else zlib.decompress(data, -15) if method == 8 else None
                if payload is None or len(payload) != uncompressed: raise ValueError(f"unsupported or corrupt ZIP entry {name}")
                temporary = args.output / (Path(name).name + ".tmp")
                temporary.write_bytes(payload); temporary.replace(args.output / Path(name).name)
                found.add(name)
            entries += 1
            if entries % 5000 == 0: print(f"scanned {entries} entries; extracted {len(found)}/{len(wanted)}", flush=True)
    missing = wanted - found
    if missing: raise FileNotFoundError(f"{len(missing)} requested GQA images missing; first={sorted(missing)[:3]}")
    print(f"extracted {len(found)} FineCops images from {entries} ZIP entries")


if __name__ == "__main__": main()
