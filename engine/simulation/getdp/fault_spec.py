#!/usr/bin/env python3
"""M88 短路/匝间故障工况口径卡 (doctest 可执行) — fault_report.json 唯一口径源

链路: run_emag_fault.py → generate_geo.py(Δ=0) → motor_mag_t.pro (M86 瞬态链复用)
      → 各工况 b_map_fault_*.pos → 退磁膝点校核 → fault_report.json

工况口径 (constants/operating_conditions.json fault_*):
- sym_sc 三相对称短路: 端口三相短接, 冲击电流以恒流源口径注入 —
  IA0=IB0=IC0 = fault_sc_current_pu × js_rated (默认 5 pu, 突发短路第一半波量级),
  考察时长 fault_sc_periods 个电周期, 转子置于直轴对齐最不利相位 (Δ=0 口径)。
- turn_short 匝间短路: fault_turnshort_phase 相部分匝 (fault_turnshort_frac)
  被短接, 短路匝环流以同向叠加口径注入该相槽 —
  IA_fault = js_rated + fault_turnshort_js_mult × js_rated (默认 1+3 倍保守),
  B/C 相保持额定。v1 披露: 槽级电流幅值叠加 (不分匝层), 环流相位假设与负载
  电流同向 (保守); 精细匝层建模列 v2。
- 退磁校核: 复用静磁链 demag_check (工作点 B_par vs 膝点线 B=μ_p·μ0·H,
  KNEE_PERMEANCE 判据), 对每工况末步 b_map 逐磁体单元校核;
  margin_pct = (B_par_min − B_knee)/|B_knee|·100; 判据 risk_fraction==0 为 PASS。
- 额定基线: 静磁 load (js_rated) 退磁基线同表披露, 供对比。

>>> spec = load_fault_spec()
>>> spec["sc_current_pu"] > 0 and 0 < spec["turnshort_frac"] < 1
True
>>> c = fault_cases(2.0e6, spec)
>>> sorted(c.keys())
['sym_sc', 'turn_short']
>>> c["sym_sc"]["IA0"] == c["sym_sc"]["IB0"] == c["sym_sc"]["IC0"]
True
>>> c["sym_sc"]["IA0"]                      # 5 pu × 2e6
10000000.0
>>> c["turn_short"]["IA0"]                  # 1× + 3× 同向叠加 (A 相)
8000000.0
>>> c["turn_short"]["IB0"], c["turn_short"]["IC0"]
(2000000.0, 2000000.0)
>>> m = demag_margin({"B_knee_T": -0.0926, "B_par_min_T": -0.05, "risk_fraction": 0.0})
>>> round(m["margin_pct"], 2), m["pass"]
(46.0, True)
>>> demag_margin({"B_knee_T": -0.0926, "B_par_min_T": -0.10, "risk_fraction": 0.3})["pass"]
False
>>> report_schema()
['params', 'cases', 'log', 'runtime_s']
>>> report_schema()[1]
'cases'
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONST = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "constants")


def _operating():
    with open(os.path.join(_CONST, "operating_conditions.json"),
              encoding="utf-8") as f:
        return json.load(f)


def load_fault_spec():
    """故障工况默认值 — 单一事实源 constants/operating_conditions.json (fault_*)。"""
    oc = _operating()
    return {"sc_current_pu": float(oc["fault_sc_current_pu"]),
            "sc_periods": float(oc["fault_sc_periods"]),
            "turnshort_frac": float(oc["fault_turnshort_frac"]),
            "turnshort_js_mult": float(oc["fault_turnshort_js_mult"]),
            "turnshort_phase": str(oc["fault_turnshort_phase"]).upper()}


def fault_cases(js_rated, spec):
    """两工况三相电流密度幅值 [A/m²] (注入 motor_mag_t.pro IA0/IB0/IC0)。

    对称短路: 三相 = pu·额定; 匝间: 故障相 = 额定+mult·额定 同向叠加。
    """
    js_sc = spec["sc_current_pu"] * js_rated
    js_ts = js_rated + spec["turnshort_js_mult"] * js_rated
    ph = spec["turnshort_phase"]
    amp = {"A": "IA0", "B": "IB0", "C": "IC0"}[ph]
    ts = {"IA0": js_rated, "IB0": js_rated, "IC0": js_rated}
    ts[amp] = js_ts
    return {
        "sym_sc": {"IA0": js_sc, "IB0": js_sc, "IC0": js_sc},
        "turn_short": ts,
    }


def sc_periods(spec):
    """对称短路考察时长 = sc_periods 个电周期 (匝间同口径, 末步校核)。"""
    return spec["sc_periods"]


def demag_margin(demag):
    """退磁裕度: margin_pct=(B_par_min−B_knee)/|B_knee|·100; risk==0 判 PASS。"""
    bk, bp = demag["B_knee_T"], demag["B_par_min_T"]
    if bk is None or bp is None or bk == 0:
        return {"margin_pct": None, "pass": False}
    return {"margin_pct": round((bp - bk) / abs(bk) * 100.0, 2),
            "pass": bool(demag.get("risk_fraction", 1.0) == 0.0)}


def sc_report_keys():
    """单工况报告键序 (fault_report.json cases[i] 内键)。"""
    return ["i_pk_Am2", "periods", "Bg1_T", "torque_Nm",
            "H_knee_Am", "B_knee_T", "B_par_min_T", "risk_fraction",
            "margin_pct", "pass"]


def report_schema():
    """fault_report.json 顶层键 (params / cases / log / runtime_s;
    baseline_rated 为可选键, --skip-baseline 时缺席)。"""
    return ["params", "cases", "log", "runtime_s"]


def report_keys_ok(rep):
    """报告顶层键校验: 必需键全在, 且只允许 baseline_rated 额外可选键。

    >>> report_keys_ok({"params": {}, "cases": [], "log": [], "runtime_s": 1.0})
    True
    >>> report_keys_ok({"params": {}, "cases": [], "log": [], "runtime_s": 1.0,
    ...                 "baseline_rated": {}})
    True
    >>> report_keys_ok({"params": {}, "cases": []})
    False
    """
    req = set(report_schema())
    opt = {"baseline_rated"}
    keys = set(rep.keys())
    return req <= keys and keys <= req | opt


if __name__ == "__main__":
    import doctest
    fails, tested = doctest.testmod().failed, doctest.testmod().attempted
    print(f"fault_spec 口径卡 doctest: {tested - fails}/{tested} PASS")
    raise SystemExit(1 if fails else 0)
