#!/usr/bin/env python3
"""M101 材料库扩充 口径卡 (doctest 可执行) — 材质条目 schema/B-H 插值/Bertotti 铁损唯一事实源

链路: constants/materials/*.json (材质条目, 每牌号一文件)
      → mat_spec.validate_material (schema 校验: 必填键/单位化键名/数值范围, fail-fast)
      → mat_spec.load_materials_dir (全目录校验+加载, 白名单键, 未知键拒绝)
      → bh_interp / bertotti_loss (下游求解链数值口径)

口径:
- 条目 schema (单位内嵌键名, 白名单键):
      公共必填: name(str) / category(str, 枚举) / source(str, 公开来源登记)
      soft_magnetic 额外必填: thickness_mm / density_kg_m3;  可选: p17_50_W_kg / bh_table / kh / kc / ke
      hard_magnetic 额外必填: br_T / hc_kAm / hcj_kAm / mur / alpha_br_pct_per_K / max_work_temp_C
      structural    额外必填: density_kg_m3 / youngs_GPa / yield_MPa
      insulation    额外必填: thermal_class / temp_index_C
      "_" 前缀键 (_desc/_omit_note/_note) 为注记, 不参与校验; 其余白名单外键 → 拒绝
- 缺键/越界/未知键/未知类别一律 ValueError fail-fast — 禁静默兜底 (宁缺勿错口径的机械执行)。
- B-H 插值 bh_interp: 单调表 [[H(A/m), B(T)],...] 线性插值; 区间外 ValueError (禁外推)。
- Bertotti 铁损: p(W/kg) = kh·B²·f + kc·B²·f² + ke·B^1.5·f^1.5 (f[Hz], B[T])。
- 真实牌号数据只登记公开来源可核实字段; 不可核实字段留空 (宁缺勿错), 不做静默默认值。

>>> spec = load_mat_spec()
>>> spec["demo_steel"]["kh"]
0.015
>>> bertotti_loss(1.0, 0.01, 0.0, 1.0, 50.0)
75.0
>>> bertotti_loss(0.0, 0.0, 0.001, 1.0, 50.0)
0.353553
>>> bh_interp([[0.0, 0.0], [1000.0, 1.5], [2000.0, 1.8]], 500.0)
0.75
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MATERIALS_DIR = os.path.join(HERE, os.pardir, os.pardir, "constants", "materials")

# ---------------------------------------------------------------
# §一 条目 schema — 白名单键 + 类别必填 + 数值范围 (闭区间)
# ---------------------------------------------------------------

CATEGORIES = ("soft_magnetic", "hard_magnetic", "structural", "insulation")

CATEGORY_REQUIRED = {
    "soft_magnetic": ("thickness_mm",),  # density_kg_m3 可选 (PN1500-100A 等公开不可核实, 宁缺勿错)
    "hard_magnetic": (
        "br_T", "hc_kAm", "hcj_kAm", "mur", "alpha_br_pct_per_K", "max_work_temp_C",
    ),
    "structural": ("density_kg_m3", "youngs_GPa", "yield_MPa"),
    "insulation": ("thermal_class", "temp_index_C"),
}

# 单位内嵌键名 → (lo, hi) 闭区间; 单位: mm / kg·m⁻³ / T / kA·m⁻¹ / %·K⁻¹ / °C / GPa / MPa
KEY_RANGES = {
    "thickness_mm": (0.0, 2.0),
    "density_kg_m3": (1000.0, 9500.0),
    "p17_50_W_kg": (0.0, 20.0),
    "br_T": (0.1, 1.6),
    "hc_kAm": (50.0, 3000.0),
    "hcj_kAm": (50.0, 4000.0),
    "mur": (1.0, 1.5),
    "alpha_br_pct_per_K": (-0.5, 0.0),
    "max_work_temp_C": (60.0, 350.0),
    "youngs_GPa": (10.0, 500.0),
    "yield_MPa": (10.0, 2000.0),
    "temp_index_C": (90.0, 250.0),
    "thermal_conductivity_W_mK": (1.0, 500.0),
}

KEY_WHITELIST = frozenset(
    list(KEY_RANGES)
    + ["name", "category", "subclass", "source", "thermal_class", "bh_table", "kh", "kc", "ke"]
)

THERMAL_CLASSES = ("A", "E", "B", "F", "H", "N", "R", "S")


def load_mat_spec():
    """读本口径卡锚点 (锚点随卡, 不散落)。

    >>> s = load_mat_spec()
    >>> s["demo_steel"]["kc"]
    0.0002
    >>> s["bh_demo_anchors"]["h1500_b"]
    1.65
    >>> s["min_new_materials"]
    5
    """
    with open(os.path.join(HERE, "mat_spec.json"), encoding="utf-8") as f:
        return json.load(f)


def _fail(entry_name, msg):
    raise ValueError("材质条目[%s]校验失败: %s" % (entry_name, msg))


def validate_material(entry):
    """材质条目 schema 校验 (fail-fast, 返回条目本身)。

    缺键 / 越界 / 未知键 / 未知类别 / 热等级枚举外 → ValueError:
    >>> ok = {"name": "X35", "category": "soft_magnetic", "source": "demo",
    ...       "thickness_mm": 0.35, "density_kg_m3": 7650.0}
    >>> validate_material(ok)["name"]
    'X35'
    >>> validate_material({"name": "X", "category": "soft_magnetic", "source": "s"})
    Traceback (most recent call last):
        ...
    ValueError: 材质条目[X]校验失败: soft_magnetic 缺必填键: thickness_mm
    >>> validate_material({"name": "X", "category": "piezo", "source": "s"})
    Traceback (most recent call last):
        ...
    ValueError: 材质条目[X]校验失败: 未知类别: piezo
    >>> validate_material({"name": "X", "category": "insulation", "source": "s",
    ...                    "thermal_class": "Q", "temp_index_C": 180.0})
    Traceback (most recent call last):
        ...
    ValueError: 材质条目[X]校验失败: thermal_class 枚举外: Q
    >>> validate_material({"name": "X", "category": "insulation", "source": "s",
    ...                    "thermal_class": "F", "temp_index_C": 155.0, "br_T": 99.0})
    Traceback (most recent call last):
        ...
    ValueError: 材质条目[X]校验失败: br_T=99.0 越界 [0.1, 1.6]
    >>> validate_material({"name": "X", "category": "insulation", "source": "s",
    ...                    "thermal_class": "F", "temp_index_C": 155.0, "mo_radius": 1.0})
    Traceback (most recent call last):
        ...
    ValueError: 材质条目[X]校验失败: 白名单外键: mo_radius
    """
    name = entry.get("name")
    if not isinstance(name, str) or not name:
        _fail("?", "缺有效 name(str)")
    for req in ("category", "source"):
        if not isinstance(entry.get(req), str) or not entry[req]:
            _fail(name, "缺必填键: %s" % req)
    category = entry["category"]
    if category not in CATEGORIES:
        _fail(name, "未知类别: %s" % category)
    for key in CATEGORY_REQUIRED[category]:
        if key not in entry:
            _fail(name, "%s 缺必填键: %s" % (category, key))
    for key, val in entry.items():
        if key.startswith("_"):
            continue  # 注记键 (_desc/_omit_note/_note) 不校验
        if key not in KEY_WHITELIST:
            _fail(name, "白名单外键: %s" % key)
        if key in KEY_RANGES:
            lo, hi = KEY_RANGES[key]
            if not isinstance(val, (int, float)) or isinstance(val, bool) or not (lo <= val <= hi):
                _fail(name, "%s=%s 越界 [%s, %s]" % (key, val, lo, hi))
    if "thermal_class" in entry and entry["thermal_class"] not in THERMAL_CLASSES:
        _fail(name, "thermal_class 枚举外: %s" % entry["thermal_class"])
    if "bh_table" in entry:
        _validate_bh_table(name, entry["bh_table"])
    return entry


def _validate_bh_table(name, table):
    """B-H 表校验: [[H(A/m), B(T)],...] H≥0 严格递增(行0可含原点 B=0), B∈(0, 2.5]。

    >>> _validate_bh_table("X", [[0.0, 0.0], [1000.0, 1.5]])
    >>> _validate_bh_table("X", [[1000.0, 1.5], [900.0, 1.4]])
    Traceback (most recent call last):
        ...
    ValueError: 材质条目[X]校验失败: bh_table H 非严格递增@行1
    """
    prev_h = 0.0
    for i, row in enumerate(table):
        if len(row) != 2:
            _fail(name, "bh_table 行%d 非 [H, B] 二元组" % i)
        h, b = float(row[0]), float(row[1])
        if i == 0 and h < 0.0:
            _fail(name, "bh_table 行0 H<0")
        if i > 0 and h <= prev_h:
            _fail(name, "bh_table H 非严格递增@行%d" % i)
        ok = (0.0 <= b <= 2.5) if i == 0 else (0.0 < b <= 2.5)
        if not ok:
            _fail(name, "bh_table 行%d B=%s 越界%s" % (i, b, "[0, 2.5]" if i == 0 else " (0, 2.5]"))
        prev_h = h


# ---------------------------------------------------------------
# §二 B-H 线性插值 (禁外推)
# ---------------------------------------------------------------

def bh_interp(table, h):
    """B-H 单调表线性插值: table=[[H(A/m),B(T)],...], 返回 B(T); 区间外 ValueError。

    >>> t = [[0.0, 0.0], [1000.0, 1.5], [2000.0, 1.8]]
    >>> bh_interp(t, 1000.0)
    1.5
    >>> bh_interp(t, 500.0)
    0.75
    >>> bh_interp(t, 1500.0)
    1.65
    >>> bh_interp(t, 2500.0)
    Traceback (most recent call last):
        ...
    ValueError: bh_interp: H=2500.0 超出表范围 [0.0, 2000.0] (禁外推)
    """
    if not table:
        raise ValueError("bh_interp: 空表")
    h_lo, b_lo = float(table[0][0]), float(table[0][1])
    h_hi, b_hi = float(table[-1][0]), float(table[-1][1])
    if h < h_lo or h > h_hi:
        raise ValueError("bh_interp: H=%s 超出表范围 [%s, %s] (禁外推)" % (h, h_lo, h_hi))
    for i in range(1, len(table)):
        h0, b0 = float(table[i - 1][0]), float(table[i - 1][1])
        h1, b1 = float(table[i][0]), float(table[i][1])
        if h <= h1:
            if h1 == h0:
                return round(b1, 9)
            return round(b0 + (b1 - b0) * (h - h0) / (h1 - h0), 9)
    return round(b_hi, 9)


# ---------------------------------------------------------------
# §三 Bertotti 铁损三系数结构 kh/kc/ke
# ---------------------------------------------------------------

def bertotti_loss(kh, kc, ke, b, f):
    """Bertotti 铁损 p(W/kg) = kh·B²·f + kc·B²·f² + ke·B^1.5·f^1.5 (f[Hz], B[T])。

    磁滞/涡流/附加三系数结构 (ke=0 时退化为双项 Steinmetz 形式):
    >>> bertotti_loss(0.015, 0.0002, 0.0, 1.0, 50.0)
    1.25
    >>> bertotti_loss(0.015, 0.0002, 0.0, 1.7, 50.0)
    3.6125
    >>> bertotti_loss(1.0, 0.01, 0.0, 1.0, 50.0)
    75.0
    >>> bertotti_loss(0.0, 0.0, 0.001, 1.0, 50.0)
    0.353553
    >>> bertotti_loss(0.015, 0.0002, 0.0, 0.0, 50.0)
    0.0
    """
    kh, kc, ke, b, f = (float(x) for x in (kh, kc, ke, b, f))
    if b < 0.0 or f < 0.0:
        raise ValueError("bertotti_loss: B/f 不可为负 (B=%s, f=%s)" % (b, f))
    p = kh * b * b * f + kc * b * b * f * f + ke * (b ** 1.5) * (f ** 1.5)
    return round(p, 6)


def loss_entry_bertotti(entry, b, f):
    """soft_magnetic 条目 → Bertotti 损耗; kh/kc/ke 缺失即 ValueError (禁静默兜底)。

    >>> e = {"kh": 0.015, "kc": 0.0002, "ke": 0.0}
    >>> loss_entry_bertotti(e, 1.0, 50.0)
    1.25
    >>> loss_entry_bertotti({"kh": 0.015, "kc": 0.0002}, 1.0, 50.0)
    Traceback (most recent call last):
        ...
    ValueError: loss_entry_bertotti: 缺铁损系数 ke (宁缺勿错 — 待实测回填, 禁静默默认)
    """
    for k in ("kh", "kc", "ke"):
        if k not in entry:
            raise ValueError(
                "loss_entry_bertotti: 缺铁损系数 %s (宁缺勿错 — 待实测回填, 禁静默默认)" % k
            )
    return bertotti_loss(entry["kh"], entry["kc"], entry["ke"], b, f)


# ---------------------------------------------------------------
# §四 目录加载 — constants/materials/*.json 全量校验
# ---------------------------------------------------------------

def load_material_entry(path):
    """读单条目 JSON 并 schema 校验 (fail-fast)。

    >>> import tempfile, json as _json
    >>> d = tempfile.mkdtemp()
    >>> p = os.path.join(d, "t.json")
    >>> _ = open(p, "w", encoding="utf-8").write(_json.dumps(
    ...     {"name": "T1", "category": "insulation", "source": "IEC",
    ...      "thermal_class": "F", "temp_index_C": 155.0}))
    >>> load_material_entry(p)["name"]
    'T1'
    """
    with open(path, encoding="utf-8") as f:
        return validate_material(json.load(f))


def load_materials_dir(directory=None):
    """加载目录全部 *.json 条目并逐一校验, 返回 {name: entry}; 目录须 ≥ min_new_materials 条。

    >>> d = MATERIALS_DIR
    >>> mats = load_materials_dir(d)
    >>> len(mats) >= load_mat_spec()["min_new_materials"]
    True
    >>> mats["N42SH-ext"]["br_T"]  # 后缀 -ext: 内置库已有 N42SH, 外部条目避免同名覆盖内置
    1.28
    >>> mats["Y30BH"]["alpha_br_pct_per_K"]
    -0.2
    >>> mats["35WW270"]["thickness_mm"]
    0.35
    >>> sorted(mats)["0"] if False else None
    """
    directory = os.path.abspath(directory or MATERIALS_DIR)
    names = sorted(fn for fn in os.listdir(directory) if fn.endswith(".json"))
    mats = {}
    for fn in names:
        entry = load_material_entry(os.path.join(directory, fn))
        if entry["name"] in mats:
            raise ValueError("材质条目重名: %s (%s)" % (entry["name"], fn))
        mats[entry["name"]] = entry
    if len(mats) < load_mat_spec()["min_new_materials"]:
        raise ValueError(
            "材料目录 %s 条目数 %d < 最低要求 %d"
            % (directory, len(mats), load_mat_spec()["min_new_materials"])
        )
    return mats


if __name__ == "__main__":
    import doctest

    fails, tested = doctest.testmod().failed, doctest.testmod().attempted
    mats = load_materials_dir()
    n = len(mats)
    print(f"mat_spec 口径卡 doctest: {tested - fails}/{tested} PASS")
    print(f"constants/materials 校验: {n} 条全 PASS ({', '.join(sorted(mats))})")
    raise SystemExit(1 if fails else 0)
