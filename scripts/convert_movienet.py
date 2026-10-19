"""一次性数据工具：把 MovieNet 官方分发格式转换为项目适配器假定的格式。

官方分发（--movienet-root，如 /cache/MovieNet）:
    list.v1.txt                     1,100 部 movie1K 的 IMDb id（每行一个）
    meta_data/meta/<tt>.json        逐电影元数据（title 内嵌年份）
    script1K.v1.zip                 script/<tt>.script（480 部）
    subtitle1K.v1.zip               subtitle/<tt>.srt（816 部）
    frames/frames/<tt>/*.jpg        关键帧（已解压，~172GB）

输出（--out，默认 data/raw/movienet，即适配器假定格式）:
    movies.csv                      movie_id,title,year,imdb_id
    keyframes/<tt> -> 符号链接       省磁盘（--copy-frames 可改为复制）
    scripts/<tt>.txt                从 zip 抽取改名
    subtitles/<tt>.srt              从 zip 抽取

用法（服务器上）:
    .venv/bin/python scripts/convert_movienet.py \
        --movienet-root /cache/MovieNet --out data/raw/movienet
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="replace")

_YEAR_TAIL_RE = re.compile(r"\(([^()]*)\)\s*$")
_YEAR_IN_RE = re.compile(r"(\d{4})")


def _parse_year(title: str) -> int | None:
    """从标题尾部的括号段提取年份——兼容 "(2015)" 与 "(TV Movie 1998)" 两种格式。"""
    m = _YEAR_TAIL_RE.search(title)
    if not m:
        return None
    y = _YEAR_IN_RE.search(m.group(1))
    return int(y.group(1)) if y else None


def _movie_ids(list_path: Path) -> list[str]:
    return [ln.strip() for ln in list_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _meta_rows(root: Path, ids: list[str]) -> tuple[list[dict], int]:
    """读取 list 内电影的 meta JSON → movies.csv 行；返回 (行, 缺失数)。"""
    rows, missing = [], 0
    for tt in ids:
        p = root / "meta_data" / "meta" / f"{tt}.json"
        if not p.exists():
            missing += 1
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        title = d.get("title") or ""
        rows.append({"movie_id": tt, "title": title,
                     "year": _parse_year(title) or "",
                     "imdb_id": tt})
    return rows, missing


def _extract_zip_members(zip_path: Path, ids: set[str], inner_dir: str,
                         out_dir: Path, rename_ext: str | None) -> int:
    """从 zip 抽取 list 内电影的文件到 out_dir/<tt>.<ext>；返回抽取数。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            p = Path(name)
            if len(p.parts) != 2 or p.parts[0] != inner_dir:
                continue
            tt = p.stem
            if tt not in ids:
                continue
            ext = rename_ext if rename_ext is not None else p.suffix
            (out_dir / f"{tt}{ext}").write_bytes(zf.read(name))
            n += 1
    return n


def _link_frames(frames_dir: Path, ids: list[str], out_dir: Path,
                 copy: bool) -> int:
    """keyframes/<tt> 指向 frames/frames/<tt>（默认符号链接，--copy-frames 复制）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for tt in ids:
        src = frames_dir / tt
        dst = out_dir / tt
        if not src.is_dir():
            continue
        if dst.exists() or dst.is_symlink():
            continue
        if copy:
            shutil.copytree(src, dst)
        else:
            dst.symlink_to(src, target_is_directory=True)
        n += 1
    return n


def convert(movienet_root: Path, out: Path, copy_frames: bool = False) -> dict:
    import pandas as pd

    ids = _movie_ids(movienet_root / "list.v1.txt")
    id_set = set(ids)
    rows, meta_missing = _meta_rows(movienet_root, ids)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["movie_id", "title", "year", "imdb_id"]).to_csv(
        out / "movies.csv", index=False)
    n_scripts = _extract_zip_members(
        movienet_root / "script1K.v1.zip", id_set, "script",
        out / "scripts", rename_ext=".txt")
    n_subs = _extract_zip_members(
        movienet_root / "subtitle1K.v1.zip", id_set, "subtitle",
        out / "subtitles", rename_ext=None)
    n_frames = _link_frames(
        movienet_root / "frames" / "frames", ids, out / "keyframes", copy_frames)
    stats = {"list": len(ids), "meta": len(rows), "meta_missing": meta_missing,
             "scripts": n_scripts, "subtitles": n_subs, "frames": n_frames}
    print("[convert] MovieNet 转换完成:", json.dumps(stats, ensure_ascii=False))
    print(f"[convert] 输出目录: {out}")
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="MovieNet 官方格式 → 适配器假定格式")
    parser.add_argument("--movienet-root", required=True, help="如 /cache/MovieNet")
    parser.add_argument("--out", default="data/raw/movienet")
    parser.add_argument("--copy-frames", action="store_true",
                        help="复制关键帧而非符号链接（默认链接，省磁盘）")
    args = parser.parse_args()
    root = Path(args.movienet_root)
    for required in ("list.v1.txt", "meta_data", "script1K.v1.zip", "subtitle1K.v1.zip"):
        if not (root / required).exists():
            print(f"[convert] 缺少 {root / required}")
            return 1
    convert(root, Path(args.out), args.copy_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
