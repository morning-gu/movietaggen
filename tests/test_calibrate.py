"""余弦→genome 量纲校准：单调有界、训练点命中、JSON 往返（无重拟合）、线性法。

brief 训练点（保序）：pred [0.6,0.7,0.8,0.9] → truth [0.02,0.10,0.35,0.60]，
保序回归恰过训练点——transform(0.8)=0.35；越界钳到端点 y 阈值（0.02 / 0.60），
训练点之间线性插值（0.75 → 0.225 手算可验）。
线性法用精确线性数据 pred [0.0,0.5,1.0] → truth [0.1,0.35,0.6]
（斜率 0.5、截距 0.1 全手算）：transform(0.8)=0.5、transform(2.0) 钳 1、
transform(-1.0) 钳 0。
"""
import json

import numpy as np
import pandas as pd
import pytest

from movietaggen.align.calibrate import Calibrator, fit_calibration

BRIEF_PAIRS = pd.DataFrame(
    {"pred": [0.6, 0.7, 0.8, 0.9], "truth": [0.02, 0.10, 0.35, 0.60]})
LINEAR_PAIRS = pd.DataFrame(
    {"pred": [0.0, 0.5, 1.0], "truth": [0.1, 0.35, 0.6]})
# 全新输入：含域内非训练点与越界值（往返一致性须在不重合输入上验证）
FRESH_INPUTS = np.array([0.33, 0.57, 0.71, 0.83, 0.97, 0.02, 1.5, -0.5])


# ---------- 保序回归：单调、有界、训练点 ----------


def test_isotonic_monotone_and_bounded():
    """brief 原样检查：5 点网格单调非降且钳 [0,1]；训练点 0.8 精确命中 0.35。"""
    cal = fit_calibration(BRIEF_PAIRS, "isotonic")
    out = cal.transform(np.array([0.55, 0.65, 0.75, 0.85, 0.95]))
    assert np.all(np.diff(out) >= 0) and out.min() >= 0 and out.max() <= 1
    assert abs(cal.transform(np.array([0.8]))[0] - 0.35) < 1e-6   # 训练点过拟合检查


def test_isotonic_interpolates_between_training_points():
    """训练点之间线性插值：0.75 位于 (0.7,0.10) 与 (0.8,0.35) 正中 → 0.225。"""
    cal = fit_calibration(BRIEF_PAIRS, "isotonic")
    assert cal.transform(np.array([0.75]))[0] == pytest.approx(0.225)


def test_isotonic_out_of_bounds_clips_to_endpoints():
    """out_of_bounds=clip 语义：低于/高于阈值域钳到首末 y 阈值（0.02 / 0.60）。"""
    cal = fit_calibration(BRIEF_PAIRS, "isotonic")
    assert cal.transform(np.array([0.0]))[0] == pytest.approx(0.02)
    assert cal.transform(np.array([1.0]))[0] == pytest.approx(0.60)
    assert cal.transform(np.array([-5.0]))[0] == pytest.approx(0.02)
    assert cal.transform(np.array([5.0]))[0] == pytest.approx(0.60)


# ---------- 输入形态 ----------


def test_transform_accepts_list_series_scalar_and_empty():
    """1-D 序列（list/Series）与数组同结果；标量 → 单元素数组；空输入 → 空数组（两方法）。"""
    iso = fit_calibration(BRIEF_PAIRS, "isotonic")
    lin = fit_calibration(LINEAR_PAIRS, "linear")
    for cal in (iso, lin):
        arr = cal.transform(np.array([0.8]))
        assert list(cal.transform([0.8])) == list(arr)
        assert list(cal.transform(pd.Series([0.8]))) == list(arr)
        assert cal.transform(0.8).shape == (1,)
        assert cal.transform(np.array([])).shape == (0,)


def test_transform_rejects_multidimensional_input():
    """>1 维输入 ValueError（静默展平会产出错位分数）。"""
    cal = fit_calibration(BRIEF_PAIRS, "isotonic")
    with pytest.raises(ValueError):
        cal.transform(np.zeros((2, 3)))


# ---------- 线性法 ----------


def test_linear_method_recovers_exact_line():
    """精确线性数据 → 斜率 0.5 / 截距 0.1；transform(0.8)=0.5；全新升序网格单调有界。"""
    cal = fit_calibration(LINEAR_PAIRS, "linear")
    out = cal.transform(np.linspace(-0.5, 1.5, 21))   # 升序全新网格（单调性须按序输入检验）
    assert np.all(np.diff(out) >= 0) and out.min() >= 0 and out.max() <= 1
    assert cal.transform(np.array([0.8]))[0] == pytest.approx(0.5)


def test_linear_clips_output_to_unit_interval():
    """线性外推越界 → 钳 [0,1]：x=2.0（1.1）→ 1.0；x=-1.0（-0.4）→ 0.0。"""
    cal = fit_calibration(LINEAR_PAIRS, "linear")
    assert cal.transform(np.array([2.0]))[0] == pytest.approx(1.0)
    assert cal.transform(np.array([-1.0]))[0] == pytest.approx(0.0)


def test_linear_negative_slope_raises():
    """负斜率反转排序、违反单调契约 → 拟合时 ValueError（宁可不校准，不静默反转）。"""
    anti = pd.DataFrame({"pred": [0.1, 0.2, 0.3], "truth": [0.9, 0.5, 0.1]})
    with pytest.raises(ValueError) as ei:
        fit_calibration(anti, "linear")
    assert "单调" in str(ei.value)


# ---------- 序列化往返 ----------


def test_to_dict_is_json_safe():
    """to_dict 只含 str / float / list[float]（无 numpy 类型），可直通 json.dumps。"""
    for method, pairs in (("isotonic", BRIEF_PAIRS), ("linear", LINEAR_PAIRS)):
        d = fit_calibration(pairs, method).to_dict()
        assert isinstance(d["method"], str) and d["method"] == method
        for value in d.values():
            if isinstance(value, list):
                assert all(type(v) is float for v in value)
            else:
                assert type(value) in (str, float)
        json.dumps(d)  # 不抛即为 JSON 安全


def test_roundtrip_isotonic_exact_on_fresh_inputs(tmp_path):
    """json.dumps → from_dict 重建（不重拟合）：全新输入上与原校准器逐位一致；文件往返同。"""
    cal = fit_calibration(BRIEF_PAIRS, "isotonic")
    d = cal.to_dict()
    assert set(d) == {"method", "x_thresholds", "y_thresholds"}   # 存阈值而非训练对——无从重拟合
    rebuilt = Calibrator.from_dict(json.loads(json.dumps(d)))
    assert np.array_equal(cal.transform(FRESH_INPUTS), rebuilt.transform(FRESH_INPUTS))
    # 文件往返：落盘再读回仍逐位一致
    path = tmp_path / "calib.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    from_file = Calibrator.from_dict(json.loads(path.read_text(encoding="utf-8")))
    assert np.array_equal(cal.transform(FRESH_INPUTS), from_file.transform(FRESH_INPUTS))


def test_from_dict_reconstructs_without_refitting():
    """手工阈值 dict（无任何训练数据可拟合）直接重建 → 插值行为由阈值唯一决定。"""
    cal = Calibrator.from_dict(
        {"method": "isotonic", "x_thresholds": [0.0, 1.0], "y_thresholds": [0.25, 0.75]})
    assert cal.transform(np.array([0.5]))[0] == pytest.approx(0.5)    # 正中点
    assert cal.transform(np.array([0.25]))[0] == pytest.approx(0.375)  # 手算 0.25+0.25·0.5
    assert cal.transform(np.array([2.0]))[0] == pytest.approx(0.75)   # 越界钳端点


def test_roundtrip_linear_exact(tmp_path):
    """线性参数往返：斜率/截距手算值精确恢复；json 与文件两种往返均逐位一致。"""
    cal = fit_calibration(LINEAR_PAIRS, "linear")
    d = cal.to_dict()
    assert set(d) == {"method", "slope", "intercept"}
    assert d["slope"] == pytest.approx(0.5)
    assert d["intercept"] == pytest.approx(0.1)
    rebuilt = Calibrator.from_dict(json.loads(json.dumps(d)))
    path = tmp_path / "calib_linear.json"
    path.write_text(json.dumps(d), encoding="utf-8")
    from_file = Calibrator.from_dict(json.loads(path.read_text(encoding="utf-8")))
    for other in (rebuilt, from_file):
        assert np.array_equal(cal.transform(FRESH_INPUTS), other.transform(FRESH_INPUTS))


# ---------- 拟合守卫 ----------


def test_fit_empty_pairs_raises():
    """空校准对 → ValueError（零样本无从拟合）。"""
    empty = pd.DataFrame({"pred": pd.Series([], dtype=float),
                          "truth": pd.Series([], dtype=float)})
    with pytest.raises(ValueError) as ei:
        fit_calibration(empty, "isotonic")
    assert "空" in str(ei.value)


def test_fit_missing_columns_raises():
    """缺 pred/truth 列 → ValueError，中文信息点名缺失列。"""
    with pytest.raises(ValueError) as ei:
        fit_calibration(pd.DataFrame({"pred": [0.5]}), "isotonic")
    assert "truth" in str(ei.value)


def test_fit_invalid_method_raises():
    """未知 method → ValueError，中文信息列出合法取值。"""
    with pytest.raises(ValueError) as ei:
        fit_calibration(BRIEF_PAIRS, "platt")
    msg = str(ei.value)
    assert "isotonic" in msg and "linear" in msg


def test_from_dict_rejects_malformed_payload():
    """缺 method / 未知 method / 缺参数 / 阈值长度不一致 / 阈值非升序 → ValueError。"""
    with pytest.raises(ValueError):
        Calibrator.from_dict({})
    with pytest.raises(ValueError):
        Calibrator.from_dict({"method": "platt"})
    with pytest.raises(ValueError):
        Calibrator.from_dict({"method": "linear"})
    with pytest.raises(ValueError):
        Calibrator.from_dict({"method": "isotonic",
                              "x_thresholds": [0.1, 0.2], "y_thresholds": [0.5]})
    with pytest.raises(ValueError):
        Calibrator.from_dict({"method": "isotonic",
                              "x_thresholds": [0.2, 0.1], "y_thresholds": [0.5, 0.6]})
