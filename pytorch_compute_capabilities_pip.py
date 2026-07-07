# /// script
# dependencies = [
#   "requests",
# ]
# ///
"""
Analyze all PyTorch 2.x versions and generate comprehensive table. Note the inclusion of 'manylinux_2_28_x86_64' is release name will only find 2.7.0 and newer (currently through 2.8.0).

Each wheel is processed one at a time: download -> cuobjdump -> record result to
a per-wheel JSON cache (cache_pip/) -> delete the wheel -> next. Reruns skip any
wheel already present in the cache, so the process is resumable and only ever
keeps a single wheel on disk at a time. The markdown table is rebuilt from the
cache after every wheel, so partial progress is never lost.
"""

from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
from typing import Any
import zipfile
import re

import requests

# Directory holding one JSON file per analyzed wheel (the resumable cache).
CACHE_DIR = Path("cache_pip")


def set_temp_root(tmpdir_arg: str | None) -> None:
    """
    Set tempfile.tempdir to ensure temp files go to disk, not tmpfs.
    Priority: arg → $TMPDIR → ./wheel_tmp (created).
    """
    if tmpdir_arg:
        tempdir = Path(tmpdir_arg)
    else:
        tmpdir_env = os.environ.get("TMPDIR")
        if tmpdir_env:
            tempdir = Path(tmpdir_env)
        else:
            tempdir = Path.cwd() / "wheel_tmp"

    tempdir.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(tempdir)
    print(f"Temp root: {tempdir}")


def _int_parts(text: str) -> list[int]:
    """
    Split a dotted string into integer parts for sorting, tolerating
    non-numeric components (e.g. "unknown", "2.0.0.post1"). Non-numeric parts
    sort as -1 so malformed/unknown values sink to the bottom instead of
    crashing the sort.
    """
    parts = []
    for p in text.split("."):
        try:
            parts.append(int(p))
        except ValueError:
            parts.append(-1)
    return parts


def get_pypi_package_info(
    package_name: str, version: str | None = None
) -> dict[str, Any]:
    """
    Fetch package information from PyPI API.

    Args:
        package_name: Name of the package (e.g., 'torch')
        version: Specific version to fetch, if None fetches latest

    Returns:
        Dictionary containing package information
    """
    if version:
        url = f"https://pypi.org/pypi/{package_name}/{version}/json"
    else:
        url = f"https://pypi.org/pypi/{package_name}/json"

    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def extract_python_version_from_filename(filename: str) -> str:
    """
    Extract Python version from wheel filename.

    Args:
        filename: Wheel filename (e.g., 'torch-2.8.0-cp313-cp313-manylinux_2_28_x86_64.whl')

    Returns:
        Python version string (e.g., '3.13')
    """
    # Wheel filename format:
    #   {name}-{version}[-{build}]-{python_tag}-{abi_tag}-{platform_tag}.whl
    # The optional build tag can shift positions, so locate the CPython tag
    # (e.g. cp313, cp39, cp313t) anywhere in the name instead of assuming index.
    m = re.search(r"cp(\d)(\d{1,2})", filename)
    if m:
        return f"{m.group(1)}.{m.group(2)}"
    return "unknown"


def get_wheel_download_links(package_name: str, version: str) -> list[dict[str, str]]:
    """
    Get wheel file download links for a specific PyTorch version.
    Only returns manylinux_2_28_x86_64 wheels.

    Args:
        package_name: Name of the package (e.g., 'torch')
        version: Version string (e.g., '2.8.0')

    Returns:
        List of dictionaries containing wheel file information
    """
    try:
        package_info = get_pypi_package_info(package_name, version)

        wheels = []
        for file_info in package_info["urls"]:
            if file_info["packagetype"] == "bdist_wheel":
                filename = file_info["filename"]

                # Only include manylinux_2_28_x86_64 wheels
                if "manylinux_2_28_x86_64" not in filename:
                    continue

                python_version = extract_python_version_from_filename(filename)

                wheel_info = {
                    "filename": filename,
                    "url": file_info["url"],
                    "size": file_info["size"],
                    "python_version": python_version,
                    "platform_tag": "manylinux_2_28_x86_64",
                }
                wheels.append(wheel_info)

        # Sort by Python version for consistent ordering
        wheels.sort(key=lambda x: x["python_version"])
        return wheels

    except requests.exceptions.RequestException as e:
        print(f"Error fetching package info: {e}")
        return []
    except KeyError as e:
        print(f"Error parsing package info: {e}")
        return []


def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format."""
    if size_bytes >= 1024**3:
        return f"{size_bytes / (1024**3):.1f} GB"
    elif size_bytes >= 1024**2:
        return f"{size_bytes / (1024**2):.1f} MB"
    elif size_bytes >= 1024:
        return f"{size_bytes / 1024:.1f} KB"
    else:
        return f"{size_bytes} B"


def print_wheel_info(wheels: list[dict[str, str]]) -> None:
    """Print wheel information in a readable format."""
    for wheel in wheels:
        size_str = format_file_size(wheel["size"])
        print(f"{wheel['filename']} ({size_str})")
        print(f"  URL: {wheel['url']}")
        print(f"  Python: {wheel['python_version']}")
        print(f"  Platform: {wheel['platform_tag']}")
        print()


def download_wheel(url: str, filename: str, download_dir: Path) -> Path:
    """
    Download a wheel file from URL.

    Args:
        url: URL to download from
        filename: Name of the file
        download_dir: Directory to save the file

    Returns:
        Path to the downloaded file
    """
    download_path = download_dir / filename

    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(download_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"\rProgress: {percent:.1f}%", end="", flush=True)

    print(f"\nDownloaded to {download_path}")
    return download_path


def extract_wheel(wheel_path: Path, extract_dir: Path) -> Path:
    """
    Extract wheel file as a zip archive.

    Args:
        wheel_path: Path to the wheel file
        extract_dir: Directory to extract to

    Returns:
        Path to the extraction directory
    """
    with zipfile.ZipFile(wheel_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    print(f"Extracted to {extract_dir}")
    return extract_dir


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
                so_path = dest_dir / target
                print(f"Extracted {target}")
                return so_path
            else:
                print(f"Warning: torch/lib/libtorch_cuda.so not found in {wheel_path}")
                return None
    except Exception as e:
        print(f"Error extracting {wheel_path}: {e}")
        return None


def get_cuda_architectures(extract_dir: Path) -> list[str]:
    """
    Run cuobjdump on libtorch_cuda.so and extract supported architectures.

    Args:
        extract_dir: Directory where wheel was extracted

    Returns:
        List of supported CUDA architectures
    """
    libtorch_path = extract_dir / "torch" / "lib" / "libtorch_cuda.so"

    if not libtorch_path.exists():
        print(f"Warning: {libtorch_path} not found")
        return []

    try:
        cuobjdump_cmd = "cuobjdump"
        # Example: cuobjdump_cmd = "singularity exec --bind /path/to/bind_dir /path/to/cuda.sif cuobjdump"
        command_raw = f"{cuobjdump_cmd} '{libtorch_path}'"

        print(f"Running: {command_raw}")
        # No check=True: cuobjdump can emit valid arch lines and still exit
        # non-zero (e.g. "Invalid ELF" on a very large libtorch_cuda.so), so we
        # parse whatever stdout it produced regardless of the return code.
        result_raw = subprocess.run(
            command_raw,
            shell=True,
            capture_output=True,
            text=True,
            executable="/bin/bash",
        )

        print(f"cuobjdump output length: {len(result_raw.stdout)} characters")

        # Let's look for lines containing 'arch' (case insensitive)
        arch_lines = []
        for line in result_raw.stdout.split("\n"):
            if "arch" in line.lower():
                arch_lines.append(line.strip())

        if arch_lines:
            # Sort and remove duplicates
            unique_archs = sorted(set(arch_lines))
            print("Found architectures:")
            for arch in unique_archs:
                print(f"  {arch}")
            return unique_archs
        else:
            # Let's see if there are any lines that might contain architecture info
            print("No lines with 'arch' found. Looking for other patterns...")

            # Look for sm_ patterns
            sm_lines = []
            for line in result_raw.stdout.split("\n"):
                if "sm_" in line.lower():
                    sm_lines.append(line.strip())

            if sm_lines:
                print("Found lines with 'sm_' pattern:")
                for line in sm_lines[:10]:  # Show first 10 matches
                    print(f"  {line}")
                return sm_lines
            else:
                print(
                    "No architecture patterns found. Showing first 20 lines of cuobjdump output:"
                )
                lines = result_raw.stdout.split("\n")
                for i, line in enumerate(lines[:20]):
                    print(f"  {i + 1}: {line}")
                return []

    except subprocess.CalledProcessError as e:
        print(f"Error running cuobjdump command: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


def analyze_first_wheel(wheels: list[dict[str, str]]) -> dict[str, Any] | None:
    """
    Download and analyze the first wheel file to extract CUDA architectures.

    Args:
        wheels: List of wheel information

    Returns:
        Dictionary with wheel info and supported architectures
    """
    if not wheels:
        print("No wheels to analyze")
        return None

    first_wheel = wheels[0]
    print(f"Analyzing first wheel: {first_wheel['filename']}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Download the wheel
        wheel_path = download_wheel(
            first_wheel["url"], first_wheel["filename"], temp_path
        )

        # Extract the wheel
        extract_dir = temp_path / "extracted"
        extract_dir.mkdir()
        extract_wheel(wheel_path, extract_dir)

        # Get CUDA architectures
        archs = get_cuda_architectures(extract_dir)

        return {"wheel_info": first_wheel, "cuda_architectures": archs}


def cache_path_for_wheel(filename: str) -> Path:
    """Return the JSON cache path for a given wheel filename."""
    return CACHE_DIR / f"{filename}.json"


def get_cached_result(filename: str) -> dict[str, Any] | None:
    """Load a previously cached analysis result for a wheel, if present."""
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
    """Persist a single wheel's analysis result to the JSON cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = cache_path_for_wheel(result["wheel_info"]["filename"])
    with open(cache_file, "w") as f:
        json.dump(result, f, indent=2)


def load_all_cached_results() -> list[dict[str, Any]]:
    """Load every cached wheel result (used to build the table)."""
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


def process_wheel(wheel: dict[str, str], package_version: str) -> dict[str, Any]:
    """
    Analyze a single wheel: return cached result if available, otherwise
    download to a temp dir, run cuobjdump, cache the result, and let the temp
    dir (and wheel) be deleted on exit. Only one wheel is ever on disk at a time.

    Args:
        wheel: Wheel information
        package_version: Version string (e.g., '2.8.0')

    Returns:
        Dictionary with wheel info and supported architectures
    """
    cached = get_cached_result(wheel["filename"])
    if cached is not None:
        archs = cached.get("cuda_architectures", [])
        print(f"↻ Cached {wheel['filename']}: {', '.join(archs) if archs else 'None'}")
        return cached

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        try:
            # Download the wheel
            wheel_path = download_wheel(wheel["url"], wheel["filename"], temp_path)

            # Extract only libtorch_cuda.so
            extract_dir = temp_path / "extracted"
            extract_dir.mkdir()
            so_path = extract_libtorch_cuda(wheel_path, extract_dir)
            if not so_path:
                print(f"✗ Could not extract libtorch_cuda.so from {wheel['filename']}")
                return {
                    "wheel_info": wheel,
                    "cuda_architectures": [],
                    "package_version": package_version,
                }

            # Get CUDA architectures from the .so file
            archs = get_cuda_architectures(extract_dir)

            # Clean up arch strings to just extract sm_XX values
            clean_archs = []
            for arch in archs:
                if "sm_" in arch:
                    # Extract just the sm_XX part
                    match = re.search(r"sm_\d+[a-z]*", arch)
                    if match:
                        clean_archs.append(match.group())

            result = {
                "wheel_info": wheel,
                "cuda_architectures": sorted(set(clean_archs)),
                "package_version": package_version,
            }

            print(f"✓ Successfully analyzed {wheel['filename']}")
            if clean_archs:
                print(f"  Architectures: {', '.join(sorted(set(clean_archs)))}")
            else:
                print("  No CUDA architectures found")

        except Exception as e:
            print(f"✗ Error analyzing {wheel['filename']}: {e}")
            # Do not cache failures, so the wheel is retried on the next run.
            return {
                "wheel_info": wheel,
                "cuda_architectures": [],
                "package_version": package_version,
            }

    # Cache only successful analyses (outside the temp dir, wheel already deleted).
    save_result_to_cache(result)
    return result


def generate_pip_table(
    package: str, version: str, results: list[dict[str, Any]]
) -> str:
    """
    Generate a markdown table in the same format as table.md.

    Args:
        package: Package name (e.g., 'torch')
        version: Version string (e.g., '2.8.0')
        results: List of analysis results

    Returns:
        Markdown table string
    """
    lines = []
    lines.append("| package | architectures |")
    lines.append("|---------|---------------|")

    for result in results:
        wheel_info = result["wheel_info"]
        archs = result["cuda_architectures"]

        # Use the actual wheel filename
        package_name = wheel_info["filename"]

        # Format architectures
        if archs:
            arch_str = ", ".join(archs)
        else:
            arch_str = ""

        lines.append(f"| {package_name} | {arch_str} |")

    return "\n".join(lines)


def get_all_pytorch_2x_versions(package_name: str = "torch") -> list[str]:
    """
    Get all PyTorch 2.x versions available on PyPI.

    Args:
        package_name: Package name (default: 'torch')

    Returns:
        List of version strings (e.g., ['2.0.0', '2.0.1', '2.1.0', ...])
    """
    try:
        package_info = get_pypi_package_info(package_name)
        all_versions = list(package_info["releases"].keys())

        # Filter for 2.x versions and sort them
        pytorch_2x_versions = []
        for version in all_versions:
            if version.startswith("2."):
                # Skip pre-release versions (rc, dev, etc.)
                if not any(marker in version for marker in ["rc", "dev", "a", "b"]):
                    pytorch_2x_versions.append(version)

        # Sort versions using semantic versioning
        pytorch_2x_versions.sort(key=_int_parts, reverse=True)  # Latest first
        return pytorch_2x_versions

    except Exception as e:
        print(f"Error getting PyTorch versions: {e}")
        return []


def save_table_to_file(table_content: str, filename: str = "table_pip.md") -> None:
    """
    Save the markdown table to a file.

    Args:
        table_content: The markdown table content
        filename: Output filename
    """
    with open(filename, "w") as f:
        f.write(table_content)
    print(f"Table saved to {filename}")


def generate_comprehensive_pip_table(all_results: list[dict[str, Any]]) -> str:
    """
    Generate a comprehensive markdown table for all versions and wheels.
    Sorted with newest versions first, then by Python version.

    Args:
        all_results: List of all analysis results across versions

    Returns:
        Markdown table string
    """
    lines = []
    lines.append("| package | architectures |")
    lines.append("|---------|---------------|")

    # Sort results: newest version first, then by python version
    def sort_key(result):
        version = result["package_version"]
        python_version = result["wheel_info"]["python_version"]

        # Parse version for proper sorting (e.g., "2.8.0" -> [2, 8, 0]);
        # tolerant of non-numeric parts like "unknown" or ".post1".
        version_parts = _int_parts(version)

        # Parse python version (e.g., "3.10" -> [3, 10])
        python_parts = _int_parts(python_version)

        # Return tuple: (negative version for desc order, python version for asc order)
        return ([-x for x in version_parts], python_parts)

    sorted_results = sorted(all_results, key=sort_key)

    for result in sorted_results:
        wheel_info = result["wheel_info"]
        archs = result["cuda_architectures"]

        # Use the actual wheel filename
        package_name = wheel_info["filename"]

        # Format architectures
        if archs:
            arch_str = ", ".join(archs)
        else:
            arch_str = ""

        lines.append(f"| {package_name} | {arch_str} |")

    return "\n".join(lines)


def main():
    """Main function to analyze all PyTorch 2.x versions and generate comprehensive table."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze PyTorch wheels from PyPI")
    parser.add_argument("--tmpdir", help="Disk-backed temp directory (default: $TMPDIR or ./wheel_tmp)")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    set_temp_root(args.tmpdir)

    package = "torch"

    # Get all PyTorch 2.x versions
    print("Fetching all PyTorch 2.x versions from PyPI...")
    versions = get_all_pytorch_2x_versions(package)

    if not versions:
        print("No PyTorch 2.x versions found!")
        return

    print(f"Found {len(versions)} PyTorch 2.x versions:")
    for i, version in enumerate(versions):
        print(f"  {i + 1:2}. {version}")
    print()

    # Count total wheels before processing
    print("Counting total wheels to be processed...")
    total_wheels_estimate = 0
    version_wheel_counts = {}

    for version in versions:
        wheels = get_wheel_download_links(package, version)
        wheel_count = len(wheels)
        version_wheel_counts[version] = wheel_count
        total_wheels_estimate += wheel_count
        print(f"  {version}: {wheel_count} wheels")

    print(f"\nTotal wheels to process: {total_wheels_estimate}")

    if total_wheels_estimate == 0:
        print("No wheels found to process!")
        return

    # How many wheels are already cached from a previous run?
    cached_count = sum(
        1
        for version in versions
        for wheel in get_wheel_download_links(package, version)
        if get_cached_result(wheel["filename"]) is not None
    )
    remaining = total_wheels_estimate - cached_count
    print(f"Already cached: {cached_count} wheels (will be skipped)")

    # Estimate download size for the remaining wheels only (~850MB per wheel).
    estimated_size_gb = (remaining * 850) / 1024
    print(f"Wheels left to download: {remaining} (~{estimated_size_gb:.1f} GB)")
    print("Each wheel is downloaded, analyzed, then deleted before the next one.")
    print()

    # Ask for confirmation, unless -y/--yes was passed or stdin is non-interactive.
    if remaining > 0 and not args.yes and sys.stdin.isatty():
        try:
            confirm = input("Do you want to proceed? (y/N): ").strip().lower()
            if confirm not in ["y", "yes"]:
                print("Aborted.")
                return
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    def rebuild_table() -> None:
        """Regenerate the markdown table from everything currently cached."""
        results = load_all_cached_results()
        if results:
            save_table_to_file(generate_comprehensive_pip_table(results))

    total_wheels = 0

    # Process each wheel one at a time, rebuilding the table from cache as we go
    # so that progress survives an interruption.
    for version_idx, version in enumerate(versions, 1):
        wheel_count = version_wheel_counts[version]

        if wheel_count == 0:
            print(f"Skipping {version} (no manylinux_2_28_x86_64 wheels)")
            continue

        print(f"\n{'=' * 80}")
        print(
            f"Processing version {version_idx}/{len(versions)}: {package} {version} ({wheel_count} wheels)"
        )
        print(f"{'=' * 80}")

        wheels = get_wheel_download_links(package, version)

        for i, wheel in enumerate(wheels, 1):
            print(f"\n--- Wheel {i}/{len(wheels)}: {wheel['filename']} ---")
            result = process_wheel(wheel, version)
            total_wheels += 1

            # Persist progress after every wheel (only if it was cached, i.e. ok).
            if get_cached_result(wheel["filename"]) is not None:
                rebuild_table()

    # Final table rebuild + summary from the full cache.
    all_results = load_all_cached_results()
    if all_results:
        print(f"\n{'=' * 80}")
        print(f"Generating comprehensive table from {len(all_results)} cached wheels...")
        print(f"Wheels visited this run: {total_wheels}")
        print(f"{'=' * 80}")

        rebuild_table()

        # Final summary
        version_counts: dict[str, int] = {}
        for result in all_results:
            version = result["package_version"]
            version_counts[version] = version_counts.get(version, 0) + 1

        print("\nFinal Summary:")
        print(f"Total PyTorch versions in cache: {len(version_counts)}")
        print(f"Total wheel files in cache: {len(all_results)}")
        for version, count in sorted(version_counts.items(), reverse=True):
            print(f"  {version}: {count} wheels")

    else:
        print("No results to save!")


if __name__ == "__main__":
    main()
