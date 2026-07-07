# /// script
# dependencies = [
#   "requests",
# ]
# ///
"""
Scrape PyTorch's self-hosted wheel index for compute capabilities.

Index wheels by CUDA variant (`cu118`, `cu121`, etc) from
https://download.pytorch.org/whl/. Each wheel is processed: download →
extract only libtorch_cuda.so → cuobjdump → per-wheel JSON cache →
delete → next. Resumable; only one wheel on disk at a time.
"""

from pathlib import Path
import json
import re
import subprocess
import sys
import tempfile
from typing import Any
from urllib.parse import urljoin, urlparse
import zipfile

import requests

BASE_INDEX = "https://download.pytorch.org/whl"
DEFAULT_VARIANTS = ["cu118", "cu121", "cu124", "cu126", "cu128"]
CACHE_DIR = Path("cache_download")


def set_temp_root(tmpdir_arg: str | None) -> None:
    """
    Set tempfile.tempdir to ensure temp files go to disk, not tmpfs.
    Priority: arg → $TMPDIR → ./wheel_tmp (created).
    """
    if tmpdir_arg:
        tempdir = Path(tmpdir_arg)
    else:
        import os
        tmpdir_env = os.environ.get("TMPDIR")
        if tmpdir_env:
            tempdir = Path(tmpdir_env)
        else:
            tempdir = Path.cwd() / "wheel_tmp"

    tempdir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tempdir)
    print(f"Temp root: {tempdir}")


def fetch_index(variant: str) -> str:
    """
    Fetch HTML index for a single CUDA variant.

    Args:
        variant: CUDA variant (e.g., 'cu124')

    Returns:
        HTML content as string
    """
    url = f"{BASE_INDEX}/{variant}/torch/"
    print(f"Fetching {url}...")
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def parse_index(html: str, index_url: str) -> list[dict[str, str]]:
    """
    Parse PEP 503 HTML index to extract wheel filenames and URLs.

    Args:
        html: HTML content
        index_url: Base URL for relative link resolution

    Returns:
        List of dicts with 'filename' and 'url' keys
    """
    wheels = []
    # Match <a href="..." ...>filename</a>
    pattern = r'<a\s+href=["\']([^"\']+)["\'][^>]*>([^<]+)</a>'
    for match in re.finditer(pattern, html):
        href, text = match.groups()
        filename = text.strip()
        # Resolve relative URL and strip fragment
        absolute_url = urljoin(index_url, href)
        # Remove #sha256=... fragment
        parsed = urlparse(absolute_url)
        url_no_fragment = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{parsed.query}".rstrip("?")
        wheels.append({"filename": filename, "url": url_no_fragment})

    return wheels


def discover_variants() -> list[str]:
    """
    Parse top-level index to auto-discover CUDA variants (cu75, cu118, etc).

    Returns:
        List of variant names
    """
    print(f"Discovering variants from {BASE_INDEX}/...")
    response = requests.get(BASE_INDEX + "/")
    response.raise_for_status()

    variants = []
    pattern = r'<a\s+href=["\']([^"\']+)/["\']'
    for match in re.finditer(pattern, response.text):
        dirname = match.group(1)
        if re.match(r"^cu\d+$", dirname):
            variants.append(dirname)

    return sorted(variants)


def filter_wheels(
    wheels: list[dict[str, str]], variant: str
) -> list[dict[str, str]]:
    """
    Filter wheels: keep torch 2.x, linux x86_64, matching variant.
    Attach python_version and package_version.

    Args:
        wheels: List of wheel dicts
        variant: CUDA variant (e.g., 'cu124')

    Returns:
        Filtered and annotated wheel dicts
    """
    filtered = []
    for wheel in wheels:
        filename = wheel["filename"]

        # Must contain torch and the variant tag
        if "torch" not in filename or f"+{variant}" not in filename:
            continue

        # Must be linux x86_64 (match both linux_x86_64 and manylinux_2_28_x86_64)
        if not ("x86_64" in filename and "linux" in filename):
            continue

        # Extract version: torch-2.x.x format
        version_match = re.search(r"^torch-(\d+\.\d+\.\d+)", filename)
        if not version_match:
            continue
        package_version = version_match.group(1)

        # Only 2.x
        if not package_version.startswith("2."):
            continue

        # Extract Python version: cpXY tag
        python_match = re.search(r"-cp(\d)(\d+)-", filename)
        if python_match:
            major, minor = python_match.groups()
            python_version = f"{major}.{minor}"
        else:
            python_version = "unknown"

        wheel["package_version"] = package_version
        wheel["python_version"] = python_version
        filtered.append(wheel)

    return filtered


def extract_libtorch_cuda(wheel_path: Path, dest_dir: Path) -> Path | None:
    """
    Extract only torch/lib/libtorch_cuda.so from wheel.

    Args:
        wheel_path: Path to .whl file
        dest_dir: Destination directory

    Returns:
        Path to libtorch_cuda.so or None if not found
    """
    try:
        with zipfile.ZipFile(wheel_path, "r") as zf:
            # Find the entry ending with torch/lib/libtorch_cuda.so
            target = None
            for name in zf.namelist():
                if name.endswith("torch/lib/libtorch_cuda.so"):
                    target = name
                    break

            if target:
                zf.extract(target, dest_dir)
                # Return the full path to the extracted .so
                so_path = dest_dir / target
                print(f"Extracted {target}")
                return so_path
            else:
                print(f"Warning: torch/lib/libtorch_cuda.so not found in {wheel_path}")
                return None
    except Exception as e:
        print(f"Error extracting {wheel_path}: {e}")
        return None


def get_cuda_architectures(so_path: Path) -> list[str]:
    """
    Run cuobjdump on libtorch_cuda.so and extract supported architectures.

    Args:
        so_path: Path to libtorch_cuda.so

    Returns:
        List of sm_XX architecture tags
    """
    if not so_path.exists():
        print(f"Warning: {so_path} not found")
        return []

    try:
        cuobjdump_cmd = "cuobjdump"
        command_raw = f"{cuobjdump_cmd} '{so_path}'"

        print(f"Running: {command_raw}")
        result = subprocess.run(
            command_raw,
            shell=True,
            capture_output=True,
            text=True,
            executable="/bin/bash",
        )

        print(f"cuobjdump output length: {len(result.stdout)} characters")

        # Extract sm_XX[a-z]* patterns
        clean_archs = []
        for line in result.stdout.split("\n"):
            matches = re.findall(r"sm_\d+[a-z]*", line)
            clean_archs.extend(matches)

        return sorted(set(clean_archs)) if clean_archs else []

    except Exception as e:
        print(f"Error running cuobjdump: {e}")
        return []


def cache_path_for_wheel(filename: str) -> Path:
    """Return the JSON cache path for a wheel filename."""
    return CACHE_DIR / f"{filename}.json"


def get_cached_result(filename: str) -> dict[str, Any] | None:
    """Load cached result if present."""
    cache_file = cache_path_for_wheel(filename)
    try:
        with open(cache_file) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, OSError) as e:
        print(f"Warning: ignoring unreadable cache {cache_file}: {e}")
        return None


def save_result_to_cache(result: dict[str, Any]) -> None:
    """Persist result to JSON cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path_for_wheel(result["wheel_info"]["filename"])
    with open(cache_file, "w") as f:
        json.dump(result, f, indent=2)


def load_all_cached_results() -> list[dict[str, Any]]:
    """Load all cached wheel results."""
    if not CACHE_DIR.is_dir():
        return []

    results = []
    for cache_file in CACHE_DIR.glob("*.json"):
        try:
            with open(cache_file) as f:
                results.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"Warning: skipping unreadable cache {cache_file}: {e}")
    return results


def process_wheel(
    wheel: dict[str, str], variant: str
) -> dict[str, Any]:
    """
    Analyze a single wheel: return cached result if available, else download,
    extract only libtorch_cuda.so, cuobjdump, cache, and return.
    Only one wheel on disk at a time.

    Args:
        wheel: Wheel dict with 'filename', 'url', etc.
        variant: CUDA variant tag

    Returns:
        Result dict with wheel_info, cuda_architectures, package_version, variant
    """
    cached = get_cached_result(wheel["filename"])
    if cached is not None:
        archs = cached.get("cuda_architectures", [])
        print(f"↻ Cached {wheel['filename']}: {', '.join(archs) if archs else 'None'}")
        return cached

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Download
            print(f"Downloading {wheel['filename']}...")
            response = requests.get(wheel["url"], stream=True)
            response.raise_for_status()

            wheel_path = temp_path / wheel["filename"]
            with open(wheel_path, "wb") as f:
                f.write(response.content)
            print(f"Downloaded to {wheel_path}")

            # Extract only libtorch_cuda.so
            so_path = extract_libtorch_cuda(wheel_path, temp_path)
            if not so_path:
                print(f"✗ Could not extract libtorch_cuda.so from {wheel['filename']}")
                return {
                    "wheel_info": wheel,
                    "cuda_architectures": [],
                    "package_version": wheel.get("package_version", ""),
                    "variant": variant,
                }

            # Cuobjdump
            archs = get_cuda_architectures(so_path)

            result = {
                "wheel_info": wheel,
                "cuda_architectures": archs,
                "package_version": wheel.get("package_version", ""),
                "variant": variant,
            }

            print(f"✓ Successfully analyzed {wheel['filename']}")
            if archs:
                print(f"  Architectures: {', '.join(archs)}")
            else:
                print("  No CUDA architectures found")

        except Exception as e:
            print(f"✗ Error analyzing {wheel['filename']}: {e}")
            return {
                "wheel_info": wheel,
                "cuda_architectures": [],
                "package_version": wheel.get("package_version", ""),
                "variant": variant,
            }

    save_result_to_cache(result)
    return result


def generate_table(results: list[dict[str, Any]]) -> str:
    """
    Generate markdown table sorted by version (desc), variant, then python version.

    Args:
        results: List of result dicts

    Returns:
        Markdown table string
    """
    lines = [
        "| package | architectures |",
        "|---------|---------------|",
    ]

    def sort_key(result):
        version = result["package_version"]
        variant = result["variant"]
        python_version = result["wheel_info"].get("python_version", "0.0")

        # Parse versions
        version_parts = tuple(int(x) for x in version.split("."))
        python_parts = tuple(int(x) for x in python_version.split("."))

        # Desc version, asc variant, asc python
        return ([-x for x in version_parts], variant, python_parts)

    sorted_results = sorted(results, key=sort_key)

    for result in sorted_results:
        wheel_info = result["wheel_info"]
        archs = result["cuda_architectures"]
        filename = wheel_info["filename"]
        arch_str = ", ".join(archs) if archs else ""
        lines.append(f"| {filename} | {arch_str} |")

    return "\n".join(lines)


def save_table(table_md: str, table_csv: str, filename_md: str = "table_download.md", filename_csv: str = "table_download.csv") -> None:
    """Save markdown and CSV tables."""
    with open(filename_md, "w") as f:
        f.write(table_md)
    with open(filename_csv, "w") as f:
        f.write(table_csv)
    print(f"Tables saved to {filename_md} and {filename_csv}")


def generate_csv(results: list[dict[str, Any]]) -> str:
    """Generate CSV table."""
    lines = ["package,architectures"]

    def sort_key(result):
        version = result["package_version"]
        variant = result["variant"]
        python_version = result["wheel_info"].get("python_version", "0.0")
        version_parts = tuple(int(x) for x in version.split("."))
        python_parts = tuple(int(x) for x in python_version.split("."))
        return ([-x for x in version_parts], variant, python_parts)

    sorted_results = sorted(results, key=sort_key)

    for result in sorted_results:
        filename = result["wheel_info"]["filename"]
        archs = result["cuda_architectures"]
        arch_str = "; ".join(archs) if archs else ""
        lines.append(f"{filename},{arch_str}")

    return "\n".join(lines)


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape PyTorch wheel index for compute capabilities"
    )
    parser.add_argument(
        "--variant",
        nargs="+",
        default=DEFAULT_VARIANTS,
        help=f"CUDA variants to process (default: {' '.join(DEFAULT_VARIANTS)}). Use 'all' to auto-discover.",
    )
    parser.add_argument(
        "--tmpdir",
        help="Disk-backed temp directory (default: $TMPDIR or ./wheel_tmp)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()

    # Set temp root
    set_temp_root(args.tmpdir)

    # Resolve variants
    if args.variant == ["all"]:
        variants = discover_variants()
        print(f"Discovered variants: {variants}")
    else:
        variants = args.variant

    if not variants:
        print("No variants to process!")
        return

    print(f"Processing variants: {', '.join(variants)}\n")

    # Estimate work
    total_wheels = 0
    variant_wheel_map = {}

    for variant in variants:
        try:
            html = fetch_index(variant)
            wheels = parse_index(html, f"{BASE_INDEX}/{variant}/torch/")
            filtered = filter_wheels(wheels, variant)
            variant_wheel_map[variant] = filtered
            total_wheels += len(filtered)
            print(f"  {variant}: {len(filtered)} wheels")
        except Exception as e:
            print(f"  {variant}: Error fetching index: {e}")
            variant_wheel_map[variant] = []

    print(f"\nTotal wheels to process: {total_wheels}")

    if total_wheels == 0:
        print("No wheels found!")
        return

    # Count cached
    cached_count = 0
    for variant_wheels in variant_wheel_map.values():
        for wheel in variant_wheels:
            if get_cached_result(wheel["filename"]) is not None:
                cached_count += 1

    remaining = total_wheels - cached_count
    print(f"Already cached: {cached_count} wheels")
    print(f"Wheels left to download: {remaining} (~{(remaining * 850) / 1024:.1f} GB)")
    print()

    # Confirm
    if remaining > 0 and not args.yes and sys.stdin.isatty():
        try:
            confirm = input("Do you want to proceed? (y/N): ").strip().lower()
            if confirm not in ["y", "yes"]:
                print("Aborted.")
                return
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    def rebuild_table():
        """Rebuild table from cache."""
        results = load_all_cached_results()
        if results:
            table_md = generate_table(results)
            table_csv = generate_csv(results)
            save_table(table_md, table_csv)

    # Process wheels
    for variant in variants:
        wheels = variant_wheel_map[variant]
        if not wheels:
            continue

        print(f"\n{'=' * 80}")
        print(f"Processing {variant} ({len(wheels)} wheels)")
        print(f"{'=' * 80}\n")

        for i, wheel in enumerate(wheels, 1):
            print(f"[{i}/{len(wheels)}] {wheel['filename']}")
            process_wheel(wheel, variant)
            rebuild_table()

    # Final summary
    all_results = load_all_cached_results()
    if all_results:
        print(f"\n{'=' * 80}")
        print(f"Done: {len(all_results)} wheels cached")
        print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
