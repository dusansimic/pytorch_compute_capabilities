# /// script
# dependencies = [
#   "natsort",
#   "packaging",
#   "parse",
#   "tqdm",
#   "pandas",
#   "tabulate",
#   "zstandard",
# ]
# ///
import argparse
import fnmatch
import glob
import json
import os
import re
import shutil
import subprocess
import tarfile
import urllib.parse
import urllib.request
from typing import List, Mapping, Optional
from natsort import natsort_keygen
import packaging.version

import pandas as pd
import parse
import tqdm

# Conda channels that ship a "pytorch" package. Each maps to its linux-64
# subdir on anaconda.org.
CHANNELS = {
    "pytorch": "https://conda.anaconda.org/pytorch/linux-64/",
    "conda-forge": "https://conda.anaconda.org/conda-forge/linux-64/",
    "anaconda": "https://conda.anaconda.org/anaconda/linux-64/",
}

# Set from --channel in main(). BASE_URL is the repodata/package download root;
# CACHE_DIR is where per-package summary.json files live (namespaced per channel
# so different channels never collide).
BASE_URL = CHANNELS["pytorch"]
CACHE_DIR = "cache"


def strip_extension(fn: str, extensions=[".tar.bz2", ".tar.gz", ".conda"]):
    for ext in extensions:
        if fn.endswith(ext):
            return fn[: -len(ext)]
    raise ValueError(f"Unexpected extension for filename: {fn}")


def _extract_so_from_tar(tf: tarfile.TarFile, pkg_cache_dir: str) -> bool:
    """Extract every *.so member of an open tar into pkg_cache_dir."""
    match = False
    for m in tf:
        libname = os.path.basename(m.name)
        if fnmatch.fnmatch(libname, "*.so"):
            tqdm.tqdm.write(f"Extracting {libname}...")
            extracted = tf.extractfile(m)
            if extracted is None:
                continue
            with open(os.path.join(pkg_cache_dir, libname), "wb") as df:
                shutil.copyfileobj(extracted, df)
            match = True
    return match


def extract_so_files(archive_fn: str, pkg_cache_dir: str) -> bool:
    """
    Extract shared libraries from a conda package archive into pkg_cache_dir.

    Handles both the legacy ``.tar.bz2``/``.tar.gz`` format and the newer
    ``.conda`` format (a zip containing a zstd-compressed ``pkg-*.tar.zst``).
    Returns True if at least one ``*.so`` was extracted.
    """
    if archive_fn.endswith(".conda"):
        import zipfile

        import zstandard

        match = False
        with zipfile.ZipFile(archive_fn) as zf:
            inner_tars = [
                n
                for n in zf.namelist()
                if n.startswith("pkg-") and n.endswith(".tar.zst")
            ]
            dctx = zstandard.ZstdDecompressor()
            for name in inner_tars:
                with zf.open(name) as comp:
                    with dctx.stream_reader(comp) as reader:
                        # Streaming tar ("r|") works on a non-seekable stream.
                        with tarfile.open(fileobj=reader, mode="r|") as tf:
                            if _extract_so_from_tar(tf, pkg_cache_dir):
                                match = True
        return match

    with tarfile.open(archive_fn, "r:*") as tf:
        match = _extract_so_from_tar(tf, pkg_cache_dir)
        if not match:
            tqdm.tqdm.write(f"{archive_fn}/*.so not found")
            with open(os.path.join(pkg_cache_dir, "filelist.txt"), "w") as f:
                f.write("\n".join(tf.getnames()))
        return match


def download_file(src, dst, force=False):
    if not force and os.path.isfile(dst):
        return dst

    src_url = urllib.parse.urljoin(BASE_URL, src)
    bar = tqdm.tqdm(
        desc=f"Downloading {src}...",
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    )

    def _update_bar(blocks_transferred, block_size, total_size):
        bar.total = total_size
        bar.update(block_size)

    urllib.request.urlretrieve(src_url, dst, _update_bar)

    bar.close()

    return dst


def get_lib_fns(raw_pkg_archive_fn) -> List[str]:
    pkg_name = strip_extension(raw_pkg_archive_fn)

    pkg_cache_dir = os.path.join(CACHE_DIR, pkg_name)
    os.makedirs(pkg_cache_dir, exist_ok=True)

    lib_fns = glob.glob(os.path.join(pkg_cache_dir, "*.so"))

    if lib_fns:
        return lib_fns

    # Else download and extract
    cache_pkg_archive_fn = os.path.join(CACHE_DIR, raw_pkg_archive_fn)

    try:
        download_file(raw_pkg_archive_fn, cache_pkg_archive_fn)
    except Exception as exc:
        tqdm.tqdm.write(str(exc))
        os.remove(cache_pkg_archive_fn)
        raise

    try:
        tqdm.tqdm.write(f"Reading archive {cache_pkg_archive_fn}...")
        extract_so_files(cache_pkg_archive_fn, pkg_cache_dir)
    except (tarfile.TarError, EOFError, OSError) as exc:
        tqdm.tqdm.write(str(exc))
        os.remove(cache_pkg_archive_fn)
        return []
    else:
        return glob.glob(os.path.join(pkg_cache_dir, "*.so"))


def get_cached_summary(pkg_archive_fn: str) -> Optional[Mapping[str, str]]:
    pkg_name = strip_extension(pkg_archive_fn)

    cache_dir = os.path.join(CACHE_DIR, pkg_name)
    summary_fn = os.path.join(cache_dir, "summary.json")

    try:
        with open(summary_fn) as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def get_summary(pkg_archive_fn) -> Mapping[str, str]:
    summary = get_cached_summary(pkg_archive_fn)
    if summary is not None:
        return summary

    pkg_name = strip_extension(pkg_archive_fn)
    architectures = set()

    lib_fns = get_lib_fns(pkg_archive_fn)

    for lib_fn in lib_fns:
        tqdm.tqdm.write(f"Reading lib {lib_fn}...")
        # Do not use check_output: cuobjdump can print valid arch lines and then
        # exit non-zero (e.g. "Invalid ELF" on very large libtorch_cuda.so). We
        # still want the architectures it managed to emit, so capture stdout
        # regardless of the return code.
        proc = subprocess.run(
            f'cuobjdump "{lib_fn}"',
            shell=True,
            capture_output=True,
            text=True,
        )
        output = proc.stdout

        lib_archs = set(m["arch"] for m in parse.findall("arch = {arch}\n", output))

        architectures.update(lib_archs)

        if not lib_archs:
            os.remove(lib_fn)

    cache_dir = os.path.join(CACHE_DIR, pkg_name)
    summary_fn = os.path.join(cache_dir, "summary.json")
    summary = {"package": pkg_name, "architectures": ", ".join(sorted(architectures))}

    # Cleanup package archive
    if architectures:
        with open(summary_fn, "w") as f:
            json.dump(summary, f)

        try:
            os.remove(os.path.join(CACHE_DIR, pkg_archive_fn))
        except FileNotFoundError:
            pass

        for fn in lib_fns:
            try:
                os.remove(fn)
            except FileNotFoundError:
                pass

    return summary


# Matches the python tag inside a conda build string in either the pytorch
# channel style ("py3.11") or the conda-forge style ("py39", "py312",
# possibly embedded, e.g. "cuda126py312h...").
_PY_TAG_RE = re.compile(r"py(\d)\.?(\d+)")


def python_version_from_build(build: str) -> Optional[packaging.version.Version]:
    """Extract the CPython version from a conda build string, or None."""
    m = _PY_TAG_RE.search(build)
    if m is None:
        return None
    return packaging.version.parse(f"{m.group(1)}.{m.group(2)}")


# See https://devguide.python.org/versions/ for supported Python versions
PYTHON_MIN_VER = packaging.version.parse("3.9")
PYTHON_MAX_VER = packaging.version.parse("4")


def output_basename(channel: str) -> str:
    """Output file stem for a channel (pytorch stays 'table' for back-compat)."""
    return "table" if channel == "pytorch" else f"table_{channel}"


def load_all_cached_summaries() -> List[Mapping[str, str]]:
    """Load every summary.json currently in CACHE_DIR (used to build the table)."""
    summaries = []
    for summary_fn in glob.glob(os.path.join(CACHE_DIR, "*", "summary.json")):
        try:
            with open(summary_fn) as f:
                summaries.append(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            tqdm.tqdm.write(f"Skipping unreadable {summary_fn}: {exc}")
    return summaries


def rebuild_table(channel: str) -> None:
    """Regenerate the markdown + csv table from everything currently cached."""
    summaries = load_all_cached_summaries()
    if not summaries:
        return

    table = pd.DataFrame(summaries)
    table = table.sort_values("package", key=natsort_keygen(), ascending=False)

    basename = output_basename(channel)
    with open(f"{basename}.md", "w") as f:
        table.to_markdown(f, tablefmt="github", index=False)
    table.to_csv(f"{basename}.csv", index=False)


def collect_pkg_archive_fns(repodata: Mapping) -> List[str]:
    """Filter repodata for CUDA-enabled pytorch packages within the Python range."""
    # Newer channels split entries across "packages" (.tar.bz2) and
    # "packages.conda" (.conda); merge both.
    packages = {}
    packages.update(repodata.get("packages", {}))
    packages.update(repodata.get("packages.conda", {}))

    pkg_archive_fns = []
    for pkg_archive_fn, p in packages.items():
        if p["name"] != "pytorch":
            continue

        python_ver = python_version_from_build(p["build"])
        if python_ver is None:
            continue

        if (PYTHON_MAX_VER < python_ver) or (python_ver < PYTHON_MIN_VER):
            continue

        if "cuda" not in p["build"]:
            continue

        pkg_archive_fns.append(pkg_archive_fn)

    return pkg_archive_fns


def main():
    parser = argparse.ArgumentParser(
        description="Report CUDA compute capabilities of PyTorch conda packages."
    )
    parser.add_argument(
        "--channel",
        choices=sorted(CHANNELS),
        default="pytorch",
        help="conda channel to scan (default: pytorch)",
    )
    args = parser.parse_args()

    # Select channel: download root + per-channel cache namespace so channels
    # never collide (pytorch keeps the top-level 'cache/' for back-compat).
    global BASE_URL, CACHE_DIR
    BASE_URL = CHANNELS[args.channel]
    CACHE_DIR = "cache" if args.channel == "pytorch" else os.path.join("cache", args.channel)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # First of all, check that cuobjdump is available
    try:
        subprocess.check_output(
            "cuobjdump --version", shell=True, stderr=subprocess.STDOUT
        ).decode("utf-8")
    except subprocess.CalledProcessError as exc:
        print(exc.cmd)
        print(exc.output.decode("utf-8"))
        print(exc)
        print("cuobjdump not found. Please install the CUDA toolkit.")
        return

    print(f"Loading repodata for channel '{args.channel}'...")
    cached_repodata_fn = download_file(
        "repodata.json", os.path.join(CACHE_DIR, "repodata.json"), force=True
    )

    with open(cached_repodata_fn) as f:
        repodata = json.load(f)

    pkg_archive_fns = collect_pkg_archive_fns(repodata)
    print(f"Found {len(pkg_archive_fns)} matching pytorch packages.")
    print()

    # Process one package at a time: download -> cuobjdump -> cache summary ->
    # delete archive + libs (get_summary handles cleanup), then rebuild the table
    # from cache so progress survives an interruption and reruns are resumable.
    for pkg_archive_fn in tqdm.tqdm(pkg_archive_fns, desc="Processing packages..."):
        if get_cached_summary(pkg_archive_fn) is not None:
            continue

        get_summary(pkg_archive_fn)
        rebuild_table(args.channel)

    # Final rebuild covers the all-cached (nothing new) case too.
    rebuild_table(args.channel)

    print("Done.")


if __name__ == "__main__":
    main()
