"""Parallel HTTP-range extraction of the FineCops subset from GQA's ZIP."""

from __future__ import annotations

import argparse
import re
import struct
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


LOCAL = b"PK\x03\x04"
SEGMENT_BYTES = 128 * 1024 * 1024
OVERLAP_BYTES = 16 * 1024 * 1024


def archive_size(url: str) -> int:
    request = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        match = re.search(r"/(\d+)$", response.headers.get("Content-Range", ""))
        if response.status != 206 or not match:
            raise RuntimeError("GQA server did not provide byte-range responses")
        return int(match.group(1))


def extract_segment(
    url: str,
    total: int,
    core_start: int,
    core_end: int,
    wanted: set[str],
    output: Path,
) -> set[str]:
    request_start = max(0, core_start - OVERLAP_BYTES)
    request_end = min(total - 1, core_end + OVERLAP_BYTES - 1)
    request = urllib.request.Request(
        url, headers={"Range": f"bytes={request_start}-{request_end}"}
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                data = response.read()
            break
        except urllib.error.HTTPError as error:
            if error.code not in (429, 500, 502, 503, 504) or attempt == 5:
                raise
            time.sleep(min(60, 2**attempt))
    if response.status != 206:
        raise RuntimeError(f"range request returned HTTP {response.status}")

    found: set[str] = set()
    cursor = 0
    while True:
        offset = data.find(LOCAL, cursor)
        if offset < 0 or offset + 30 > len(data):
            break
        try:
            _, flags, method, _, _, _, compressed, uncompressed, name_len, extra_len = struct.unpack_from(
                "<5H3L2H", data, offset + 4
            )
        except struct.error:
            break
        header_end = offset + 30 + name_len + extra_len
        if header_end > len(data):
            break
        try:
            name = data[offset + 30 : offset + 30 + name_len].decode("utf-8")
        except UnicodeDecodeError:
            cursor = offset + 4
            continue
        payload_end = header_end + compressed
        absolute_offset = request_start + offset
        valid = (
            name.startswith("images/")
            and method in (0, 8)
            and not flags & 0x08
            and compressed <= len(data)
            and payload_end <= len(data)
        )
        if not valid:
            cursor = offset + 4
            continue
        if core_start <= absolute_offset < core_end and name in wanted:
            destination = output / Path(name).name
            if not destination.exists():
                compressed_data = data[header_end:payload_end]
                payload = (
                    compressed_data
                    if method == 0
                    else zlib.decompress(compressed_data, -15)
                )
                if len(payload) != uncompressed:
                    raise ValueError(f"corrupt ZIP entry {name}")
                temporary = destination.with_suffix(destination.suffix + ".tmp")
                temporary.write_bytes(payload)
                temporary.replace(destination)
            found.add(name)
        cursor = payload_end
    return found


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--list", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    wanted = {f"images/{name}" for name in args.list.read_text().splitlines() if name}
    args.output.mkdir(parents=True, exist_ok=True)
    total = archive_size(args.url)
    segments = [
        (start, min(start + SEGMENT_BYTES, total))
        for start in range(0, total, SEGMENT_BYTES)
    ]
    print(
        f"parallel extraction: {len(wanted)} targets, {len(segments)} ranges, "
        f"{args.workers} workers, archive={total / 1e9:.1f}GB",
        flush=True,
    )
    found = {
        f"images/{path.name}"
        for path in args.output.glob("*.jpg")
        if f"images/{path.name}" in wanted
    }
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(extract_segment, args.url, total, start, end, wanted, args.output): (start, end)
            for start, end in segments
        }
        completed = 0
        attempts: dict[tuple[int, int], int] = {}
        while pending:
            future = next(as_completed(pending))
            start, end = pending.pop(future)
            try:
                found.update(future.result())
                completed += 1
            except Exception as error:
                attempts[(start, end)] = attempts.get((start, end), 0) + 1
                if attempts[(start, end)] > 12:
                    raise
                print(f"retrying range {start}-{end} after {error}", flush=True)
                pending[pool.submit(extract_segment, args.url, total, start, end, wanted, args.output)] = (start, end)
                continue
            if completed % 4 == 0 or len(found) == len(wanted):
                print(f"completed {completed}/{len(segments)} ranges; extracted {len(found)}/{len(wanted)}", flush=True)

    missing = wanted - found
    if missing:
        raise FileNotFoundError(f"{len(missing)} requested GQA images missing; first={sorted(missing)[:3]}")
    print(f"extracted {len(found)} FineCops images", flush=True)


if __name__ == "__main__":
    main()
