#!/usr/bin/env python3
"""
constants/ 全局常量统一加载器 — 单一事实源

设计约定 (goal 硬约束):
- 所有常量统一定义在 motor_workbench/constants/ 目录, 分模块文件:
  physics.json(物理) materials.json(材料) motor_design.json(默认设计)
  operating_conditions.json(工况/求解默认)
- 代码内禁止重复字面量; 缺字段 fail-fast (KeyError), 禁止静默 0
- run_*.py 无 --params 时加载 motor_design.json 作为默认产品; 有则覆盖
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
# getdp/ → simulation/ → motor_workbench/constants/
CONST_DIR = os.path.join(os.path.dirname(os.path.dirname(HERE)), "constants")


def load(name):
    """读 constants/<name>.json; 文件缺失/JSON坏 → 异常(fail-fast)。"""
    path = os.path.join(CONST_DIR, name)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def physics():
    return load("physics.json")


def materials():
    return load("materials.json")


def ac_loss():
    """读 ac_loss.json (M92 交流铜损 Dowell 口径白名单)。"""
    return load("ac_loss.json")


def cooling():
    """读 cooling.json (M92 冷却拓扑枚举→LPTN 等效对流)。"""
    return load("cooling.json")


def concept_design():
    """读 concept_design.json (M92 概念设计解析正向流白名单)。"""
    return load("concept_design.json")


def fmu_design():
    """读 fmu_design.json (M97 ROM/FMU 导出白名单)。"""
    return load("fmu_design.json")


def drive_design():
    """读 drive_design.json (M96 驱动系统 dq+FOC+工况循环白名单)。"""
    return load("drive_design.json")


def map_design():
    """读 map_design.json (M98 效率 MAP 网格/MTPA/弱磁/等效率线白名单)。"""
    return load("map_design.json")


def fluxmap_design():
    """读 fluxmap_design.json (M99 磁链 MAP-ROM 网格/LUT/自检白名单)。"""
    return load("fluxmap_design.json")


def drive_cycle(name):
    """读 constants/cycles/<name>.json (WLTC/CLTC 公开标准工况数据表, fail-fast)。"""
    return load(os.path.join("cycles", name + ".json"))


def operating():
    return load("operating_conditions.json")


def default_design():
    """默认电机设计 (含派生量 R_rm / r_gap_mid)。"""
    d = load("motor_design.json")
    d = {k: v for k, v in d.items() if not k.startswith("_")}
    d["R_rm"] = d["R_ri"] + d["mag_t"]
    d["r_gap_mid"] = 0.5 * (d["R_si"] + d["R_rm"])
    return d


def merged_design(overrides=None):
    """默认设计 + 材料注入 + 外部参数覆盖 (UI/CLI --params)。未知键 fail-fast。

    材料字段 (Br/mur_mag/Hc0) 单一事实源=materials.json, 注入进设计 dict
    供下游 P["Br"] 等引用 — 设计文件不重复存材料值。
    """
    d = default_design()
    mag = materials()["magnet_N35UH"]
    d["Br"] = mag["Br_20"]
    d["mur_mag"] = mag["mur"]
    d["Hc0"] = mag["Hc0"]
    # M100 温变磁体: α_Br 随材料注入 (缺省 0=常温不变, 行为向后兼容)
    d["ALPHA_BR"] = float(mag.get("ALPHA_BR", 0.0))
    for k, v in (overrides or {}).items():
        if k not in d:
            raise KeyError(f"未知设计参数: {k} (合法键见 constants/motor_design.json)")
        d[k] = v
    d["R_rm"] = d["R_ri"] + d["mag_t"]
    d["r_gap_mid"] = 0.5 * (d["R_si"] + d["R_rm"])
    return d


if __name__ == "__main__":
    # 自检: 四文件可加载且派生量正确
    p = physics()
    m = materials()
    d = default_design()
    o = operating()
    assert abs(d["R_rm"] - 0.029) < 1e-12, d["R_rm"]
    assert abs(p["MU0"] - 4 * 3.141592653589793e-7) < 1e-18
    assert m["magnet_N35UH"]["Hc0"] == 8.804e5
    assert o["progress_check_interval_s"] == 120
    print("constants OK:",
          "physics=%d materials=%d design=%d operating=%d keys" % (
              len(p), sum(1 for k in m if not k.startswith("_")),
              len(d), len(o)))
