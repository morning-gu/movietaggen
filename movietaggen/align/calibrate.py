"""余弦 → genome 量纲校准：把映射聚合的原始余弦分数换算到 genome 相关度 [0,1] 尺度。

两段式管道的公平性铰链：`mapping.aggregate_scores` 产出的 score 是聚合余弦
（量纲随嵌入空间漂移），而 genome 相关度是长尾 [0,1]——评估对比前必须校准，
否则消融轴 9（余弦口径 vs genome 口径）不可比。本模块提供两种单调校准：

- isotonic：sklearn ``IsotonicRegression(out_of_bounds="clip")`` 拟合
  (pred, truth)，拟合后只保留阈值数组（X_thresholds_ / y_thresholds_），
  transform 用 np.interp 线性插值——与 sklearn 对象的 transform 逐位一致，
  反序列化因此无需重拟合。
- linear：``np.polyfit(pred, truth, 1)`` 一元线性回归，transform =
  clip(斜率·x + 截距, 0, 1)。

**分割职责（重要）**：本模块只负责"在给定的 pairs 上拟合"，绝不检查这些
pairs 来自哪个分割——训练/测试分割的防泄露由调用方接线（Task 13 的拟合
脚本必须只喂 TRAIN 分割对，并在接线处调 eval.protocol 的 assert_no_leakage；
校准拟合集泄露会让消融轴 9 的对比失效）。

单调性契约：任何 Calibrator.transform 输出对输入单调非降。保序回归天然
满足；线性法在构造时强制斜率 ≥ 0（负斜率会反转排序，等价于用方向错误的
校准污染对比，宁可在拟合时大声失败也不静默放行）。

序列化契约：to_dict 产出的 dict 仅含 str / float / list[float]（无 numpy
类型），可直通 json.dumps；from_dict 按存储参数重建、绝不重新拟合，重建后
的 transform 与原校准器逐位一致。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

# 复用 config 的合法校准方法常量（控制器所称 config.VALID_CALIB 在仓库中的实际名字）
from movietaggen.config import VALID_CALIB_METHODS


class Calibrator:
    """已拟合的校准器：method + 拟合参数，transform 单调非降且输出钳 [0,1]。

    正常入口是 fit_calibration（拟合）或 from_dict（反序列化）；直接构造需
    传齐 params（isotonic: x_thresholds/y_thresholds；linear: slope/intercept），
    构造即校验（非法参数在此大声失败，而非等到 transform 产出垃圾）。
    """

    def __init__(self, method: str, params: dict):
        if method not in VALID_CALIB_METHODS:
            raise ValueError(f"method 必须是 {VALID_CALIB_METHODS}，得到 {method!r}")
        self.method = method
        if method == "isotonic":
            missing = [k for k in ("x_thresholds", "y_thresholds") if k not in params]
            if missing:
                raise ValueError(f"isotonic 参数缺少键 {missing}（需 x_thresholds、y_thresholds）")
            x = _to_float_array(params["x_thresholds"], "x_thresholds")
            y = _to_float_array(params["y_thresholds"], "y_thresholds")
            if x.size == 0:
                raise ValueError("isotonic 阈值数组为空，无法插值")
            if x.size != y.size:
                raise ValueError(
                    f"x_thresholds 长度 {x.size} 与 y_thresholds 长度 {y.size} 不一致")
            # np.interp 产出单调非降的前提：x 升序且 y 非降（sklearn 拟合产物天然满足）
            if np.any(np.diff(x) < 0) or np.any(np.diff(y) < 0):
                raise ValueError("isotonic 阈值须非降序（单调插值的前提）")
            self._x_thresholds = x
            self._y_thresholds = y
        else:
            missing = [k for k in ("slope", "intercept") if k not in params]
            if missing:
                raise ValueError(f"linear 参数缺少键 {missing}（需 slope、intercept）")
            slope = float(params["slope"])
            intercept = float(params["intercept"])
            if slope < 0:
                raise ValueError(
                    f"linear 校准斜率 {slope} < 0：负斜率反转排序、违反单调契约"
                    "（说明 pred 与 truth 负相关，此校准无意义）")
            self._slope = slope
            self._intercept = intercept

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """分数 → 校准分（单调非降，输出钳 [0,1]）。

        - 主契约输入是 1-D 数组；list、Series、标量亦接受，标量返回单元素数组。
        - 空输入 → 空数组。
        - >1 维输入 ValueError（静默展平会产出错位分数）。
        """
        arr = np.asarray(scores, dtype=float)
        if arr.ndim > 1:
            raise ValueError(f"scores 须为 1-D 数组或标量，得到 {arr.ndim}-D")
        flat = np.atleast_1d(arr)
        if self.method == "isotonic":
            out = np.interp(flat, self._x_thresholds, self._y_thresholds)
        else:
            out = self._slope * flat + self._intercept
        return np.clip(out, 0.0, 1.0)

    def to_dict(self) -> dict:
        """序列化为 JSON 安全 dict（str / float / list[float]，无 numpy 类型）。"""
        if self.method == "isotonic":
            return {
                "method": self.method,
                "x_thresholds": [float(v) for v in self._x_thresholds],
                "y_thresholds": [float(v) for v in self._y_thresholds],
            }
        return {"method": self.method, "slope": self._slope, "intercept": self._intercept}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        """按存储参数重建（绝不重新拟合）——与 to_dict 严格互逆。"""
        if not isinstance(d, dict):
            raise ValueError(f"from_dict 需要 dict，得到 {type(d).__name__}")
        if "method" not in d:
            raise ValueError("from_dict 缺少 method 键")
        params = {k: v for k, v in d.items() if k != "method"}
        return cls(d["method"], params)


def _to_float_array(value, name: str) -> np.ndarray:
    """任意序列 → float64 数组；不可转换时 ValueError 点名列名。"""
    try:
        return np.asarray([float(v) for v in value], dtype=float)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{name} 须为数值序列：{e}") from e


def fit_calibration(pairs: pd.DataFrame, method: str) -> Calibrator:
    """在 (pred, truth) 校准对上拟合 Calibrator。

    - pairs 须含列 pred、truth（float）。**调用方保证 pairs 仅来自训练分割**：
      本模块不查分割，防泄露接线（assert_no_leakage）是调用方（Task 13/18）
      的职责。
    - method ∈ isotonic | linear（VALID_CALIB_METHODS），其他取值 ValueError。
    - 空 pairs → ValueError（零样本无从拟合校准）。
    """
    if method not in VALID_CALIB_METHODS:
        raise ValueError(f"method 必须是 {VALID_CALIB_METHODS}，得到 {method!r}")
    missing = [c for c in ("pred", "truth") if c not in pairs.columns]
    if missing:
        raise ValueError(f"pairs 缺少列 {missing}，需要 pred、truth 两列")
    if len(pairs) == 0:
        raise ValueError("校准对为空：不能在零样本上拟合校准器（请检查训练分割是否为空）")
    pred = pairs["pred"].to_numpy(dtype=float)
    truth = pairs["truth"].to_numpy(dtype=float)
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip").fit(pred, truth)
        params = {
            "x_thresholds": [float(v) for v in iso.X_thresholds_],
            "y_thresholds": [float(v) for v in iso.y_thresholds_],
        }
    else:
        slope, intercept = np.polyfit(pred, truth, 1)
        params = {"slope": float(slope), "intercept": float(intercept)}
    return Calibrator(method, params)
