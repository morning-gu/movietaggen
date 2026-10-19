"""阶段产物装载、配置快照与跨实验借用（04/05/06/07 脚本共用）。

save_config_snapshot：A 的 04 脚本行为提升为模块函数——配置快照随结果落盘，
结果目录自包含、可复现；05/06 直接复用。

load_stage 跨实验借用（spec 的实验复用模式）：input_exp 非空时把其它实验的
同名 stage 目录物理复制到本实验目录下（物理副本保证本实验目录自包含、可整体
打包/删除），并写 PROVENANCE.json 记录来源与逐文件 sha256——审计可追溯，
源实验此后被删/被改不影响已固化的借用产物（有意不做软链接/自动同步）。
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

# stage 目录 → 产出它的脚本（目录缺失时的报错信息里给出补救命令）
_STAGE_PRODUCERS = {
    "stage-gen": "python scripts/04_run_generation.py --config <配置>",
    "stage-align": "python scripts/05_run_alignment.py --config <配置>",
    "stage-audit": "python scripts/06_run_audit.py --config <配置>",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def save_config_snapshot(config_path: str | Path, paths) -> Path | None:
    """把配置 YAML 快照进结果目录（config.yaml）。源不存在 → 静默跳过返回 None。"""
    src = Path(config_path)
    if not src.exists():
        return None
    paths.results.mkdir(parents=True, exist_ok=True)
    dest = paths.results / "config.yaml"
    shutil.copy(src, dest)
    return dest


def load_stage(paths, stage: str, input_exp: str | None) -> Path:
    """返回本实验的 <stage> 目录；input_exp 非空时从该实验物理复制并写 PROVENANCE.json。

    - input_exp 为 None：本实验目录存在即返回；缺失报错（信息含补救命令）。
    - input_exp 为字符串：源 = results/<input_exp>/<stage>/，复制到本实验同名
      stage 目录（复制前清空本地，保证与源一致），PROVENANCE.json 落在目录内。
    """
    local = paths.results / stage
    if input_exp is None:
        if not local.exists():
            hint = _STAGE_PRODUCERS.get(stage, "对应阶段的生成脚本")
            raise FileNotFoundError(
                f"阶段产物缺失: {local}（stage={stage!r}）。"
                f"先运行生成命令: {hint}，或在配置中设置该阶段的 input_exp 借用其它实验。")
        return local

    if input_exp == paths.results.name:
        raise ValueError(
            f"不能借用本实验自己的产物: input_exp={input_exp!r}（会导致复制源即目标）")
    source = paths.root / "results" / input_exp / stage
    if not source.exists():
        raise FileNotFoundError(
            f"借用源不存在: {source}（input_exp={input_exp!r}, stage={stage!r}）。"
            "确认 results/ 下存在该实验的该阶段产物。")
    if local.exists():
        shutil.rmtree(local)   # 清空后复制：本地不残留与借用源不一致的旧文件
    shutil.copytree(source, local, dirs_exist_ok=True)
    files = [{"name": p.relative_to(source).as_posix(), "sha256": _sha256(p)}
             for p in sorted(source.rglob("*")) if p.is_file()]
    provenance = {"source_exp": input_exp, "stage": stage, "files": files,
                  "copied_at": datetime.now().isoformat(timespec="seconds")}
    (local / "PROVENANCE.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=1), encoding="utf-8")
    return local
