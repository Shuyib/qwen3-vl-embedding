#!/usr/bin/env python3
"""
Benchmark: extension-based vs Magika file type detection.

Answers the question: how many indexable files does extension-only miss?

Usage:
    python benchmark_file_detection.py /path/to/directory
    python benchmark_file_detection.py /path/to/directory --show-missed
    python benchmark_file_detection.py /path/to/directory --compare-file-cmd
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import NamedTuple

from src.launcher.file_type_detector import (
    AutoFileTypeDetector,
    ExtensionFileTypeDetector,
    FileTypeInfo,
)


class FileResult(NamedTuple):
    path: Path
    info: FileTypeInfo


def scan(directory: Path, detector, recursive: bool = True) -> list[FileResult]:
    pattern = "**/*" if recursive else "*"
    results = []
    for p in directory.glob(pattern):
        if p.is_file():
            results.append(FileResult(p, detector.detect(p)))
    return results


def run_unix_file_cmd(paths: list[Path], timeout: int = 60) -> dict[Path, str]:
    """Run `file --brief` on each path. Returns path -> description mapping."""
    if not paths:
        return {}

    results: dict[Path, str] = {}
    batch: list[Path] = []
    batch_chars = 0
    max_batch_chars = 100_000

    def flush_batch() -> bool:
        nonlocal batch, batch_chars
        if not batch:
            return True

        cmd = ["file", "--brief"] + [str(p) for p in batch]
        try:
            out = subprocess.check_output(cmd, timeout=timeout, text=True)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

        results.update({p: line.strip() for p, line in zip(batch, out.splitlines())})
        batch = []
        batch_chars = 0
        return True

    for path in paths:
        path_chars = len(str(path)) + 1
        if batch and batch_chars + path_chars > max_batch_chars:
            if not flush_batch():
                return results
        batch.append(path)
        batch_chars += path_chars

    flush_batch()
    return results


def print_section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark file type detection strategies")
    parser.add_argument("directory", type=Path, help="Directory to scan")
    parser.add_argument("--no-recursive", action="store_true", help="Do not recurse into subdirectories")
    parser.add_argument("--show-missed", action="store_true", help="Print files extension-only misses")
    parser.add_argument("--compare-file-cmd", action="store_true", help="Also benchmark the Unix `file` command")
    parser.add_argument("--limit", type=int, default=20, help="Max missed-file examples to show (default: 20)")
    args = parser.parse_args()

    if not args.directory.is_dir():
        sys.exit(f"Not a directory: {args.directory}")

    recursive = not args.no_recursive

    # --- Extension detector ---
    ext_detector = ExtensionFileTypeDetector()
    t0 = time.perf_counter()
    ext_results = scan(args.directory, ext_detector, recursive)
    ext_time = time.perf_counter() - t0

    ext_supported = [r for r in ext_results if r.info.is_supported]
    ext_paths = {r.path for r in ext_supported}

    print_section("Extension-based detection")
    total = len(ext_results)
    print(f"  Total files scanned : {total:,}")
    print(f"  Indexed (supported) : {len(ext_supported):,}  ({100 * len(ext_supported) / total:.1f}%)")
    print(f"    text files        : {sum(1 for r in ext_supported if r.info.launcher_type == 'text'):,}")
    print(f"    image files       : {sum(1 for r in ext_supported if r.info.launcher_type == 'image'):,}")
    print(f"  Skipped (no match)  : {total - len(ext_supported):,}")
    print(f"  Time                : {ext_time * 1000:.1f} ms")

    # --- Magika / Auto detector ---
    try:
        auto_detector = AutoFileTypeDetector(prefer_magika=True)
        magika_available = auto_detector._magika_detector is not None
    except Exception:
        magika_available = False

    if magika_available:
        t0 = time.perf_counter()
        auto_results = scan(args.directory, auto_detector, recursive)
        auto_time = time.perf_counter() - t0

        auto_supported = [r for r in auto_results if r.info.is_supported]
        auto_paths = {r.path for r in auto_supported}

        gained_paths = auto_paths - ext_paths
        lost_paths = ext_paths - auto_paths  # Magika downgraded something ext found

        print_section("Magika (content-based) detection")
        print(f"  Indexed (supported) : {len(auto_supported):,}  ({100 * len(auto_supported) / total:.1f}%)")
        print(f"    text files        : {sum(1 for r in auto_supported if r.info.launcher_type == 'text'):,}")
        print(f"    image files       : {sum(1 for r in auto_supported if r.info.launcher_type == 'image'):,}")
        print(f"  Time                : {auto_time * 1000:.1f} ms")

        print_section("Delta (Magika vs extension-only)")
        gain_pct = 100 * len(gained_paths) / len(ext_supported) if ext_supported else 0.0
        print(f"  Additional files found by Magika : +{len(gained_paths):,}  (+{gain_pct:.1f}% relative to extension baseline)")
        print(f"  Files Magika did NOT index       : -{len(lost_paths):,}  (extension found them, Magika skipped)")

        if gained_paths and (args.show_missed or len(gained_paths) <= args.limit):
            print(f"\n  Sample gained files (first {args.limit}):")
            gained_info = {r.path: r.info for r in auto_results if r.path in gained_paths}
            for p in sorted(gained_paths)[: args.limit]:
                info = gained_info[p]
                rel = p.relative_to(args.directory)
                print(f"    [{info.label:20s}] {rel}  (score={info.score:.2f})" if info.score else f"    [{info.label:20s}] {rel}")

        if lost_paths:
            print(f"\n  Files extension found but Magika skipped (first {args.limit}):")
            for p in sorted(lost_paths)[: args.limit]:
                rel = p.relative_to(args.directory)
                ext_info = {r.path: r.info for r in ext_results}[p]
                auto_info = {r.path: r.info for r in auto_results}[p]
                print(f"    ext={ext_info.launcher_type:6s} -> magika={auto_info.launcher_type:10s}  {rel}")
    else:
        print_section("Magika not installed")
        print("  Install with: uv pip install magika")
        print("  Re-run to see the delta comparison.")

    # --- Unix `file` command comparison ---
    if args.compare_file_cmd:
        print_section("`file` command (Unix magic bytes heuristic)")
        all_paths = [r.path for r in ext_results]
        t0 = time.perf_counter()
        file_cmd_results = run_unix_file_cmd(all_paths)
        file_cmd_time = time.perf_counter() - t0

        if file_cmd_results:
            text_by_file_cmd = {
                p for p, desc in file_cmd_results.items()
                if any(kw in desc.lower() for kw in ("text", "script", "source", "python", "ruby", "shell"))
            }
            image_by_file_cmd = {
                p for p, desc in file_cmd_results.items()
                if any(kw in desc.lower() for kw in ("image", "jpeg", "png", "gif", "bitmap", "svg"))
            }
            print(f"  Text-like (heuristic) : {len(text_by_file_cmd):,}")
            print(f"  Image-like (heuristic): {len(image_by_file_cmd):,}")
            print(f"  Time                  : {file_cmd_time * 1000:.1f} ms  (subprocess overhead included)")
            if magika_available:
                file_only = text_by_file_cmd - ext_paths
                print(f"  `file` finds that extension misses: {len(file_only):,}")
                magika_but_not_file = gained_paths - text_by_file_cmd  # type: ignore[name-defined]
                print(f"  Magika finds that `file` misses  : {len(magika_but_not_file):,}")
        else:
            print("  `file` command not available or timed out.")

    print_section("Interpretation")
    print("  A large +N in 'additional files found by Magika' means content-based")
    print("  detection gives meaningfully better coverage for your directory.")
    print("  Run on a real dev repo (Linux kernel, CPython, your own projects)")
    print("  to get an honest signal. Generic doc directories show less gain.")


if __name__ == "__main__":
    main()
