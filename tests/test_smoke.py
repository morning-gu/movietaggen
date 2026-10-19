"""端到端冒烟集成测试：真实跑通 00→08 全链（mock + 合成数据，零网络零 key）。

合成数据目录（data/synthetic/）与结果（results/）均被 gitignore——本测试
每次真跑全链，mock 配置下约数秒；同时兼作 run_smoke.py 的回归测试。
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture()
def repo_root() -> Path:
    """仓库根（tests/ 的上一级）——run_smoke 以相对路径读 configs/smoke.yaml。"""
    return Path(__file__).resolve().parent.parent


def test_smoke_end_to_end(repo_root, monkeypatch):
    monkeypatch.chdir(repo_root)
    from scripts.run_smoke import main
    assert main() == 0
