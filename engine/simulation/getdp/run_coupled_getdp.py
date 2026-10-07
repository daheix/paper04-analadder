#!/usr/bin/env python3
"""电磁-热耦合迭代 + 效率 Map (开源 GetDP 链, MOT-0009 / MOT-0010)

〔MOT-0009 耦合闭环〕额定工况 (js=2e6 A/m², δ=90°, 3000 rpm) 单点强耦合:
    NdFeB 可逆温度系数 α_Br ≈ −0.12%/K: Hc(T) = Hc0·(1+α_Br·(T−20))
    铜电阻温度系数 α_Cu = 3.93e-3/K (IEC 60028): R_ph(T) = R20·(1+α_Cu·(T−20))
    迭代: GetDP(Hc(T_m)) → B 场 → 损耗(Steinmetz 铁耗+铜耗) → LPTN → (T_w, T_m) → 下一轮
    收敛判据 |ΔT_magnet| < 0.3 K 且 |ΔT_winding| < 0.3 K

〔MOT-0010 效率 Map〕电流扫掠 js ∈ {1,2,3,4}×1e6 A/m² (δ=90°):
    T_avg = 1.5·p·λ₁·i_pk·sinδ (公式法; vs FEM 三口径互证 1.1% 内, analytic_verify)
    P_out = T_avg·ω_mech (ω=2π·n/60=314.16 rad/s @3000rpm)
    P_cu = 3·(i_pk/√2)²·R_ph(T_cu), P_fe = 定子 Steinmetz (该电流点 GetDP bmap)
    η = P_out/(P_out+P_cu+P_fe); 机械损/风摩/杂散损未计 (SPM 小功率可忽略量级)
"""
import json
import math
import os

import numpy as np

from run_emag_getdp import GetDPMotorDriver, P, js_to_i
from run_thermal_getdp import thermal_lptn, SPEED_RPM, HERE

from load_constants import materials as _materials_fn
_mat = _materials_fn()
HC0 = _mat["magnet_N35UH"]["Hc0"]   # A/m, constants统一管理
ABR = -1.2e-3          # NdFeB Br 可逆温度系数 /K
JS_PK = 2.0e6          # 额定槽电流密度峰值 A/m²
LAM1 = 1.1339          # Wb, 能量法 λ₁ (torque_delta_sweep 幅值法, C++ 互证)
MU0 = 4e-7 * math.pi


def _gap_bg1(drv, bmap, n=720):
    """气隙 Bn 谱: Bg1 (k=p) 与极均 (|θ|<33.75°, 口径=run_emag_getdp.py:204)。
    注意 gap_circle 返回 th 单位是"度" (np.degrees), 谱计算须转弧度。"""
    th_deg, bn, _ = drv.gap_circle(bmap, n=n)
    th = np.radians(th_deg)
    Bk = {}
    for k in (P["p"],):
        c = float(np.sum(bn * np.cos(k * th)))
        s = float(np.sum(bn * np.sin(k * th)))
        Bk[k] = 2.0 / n * math.hypot(c, s)
    d0 = np.abs(th_deg)
    pole = float(bn[d0 < P["mag_half_deg"]].mean())
    return Bk[P["p"]], pole


def coupled(h_conv=15.0, max_iter=6, tol=0.3, tag="cpl"):
    """额定工况电磁-热耦合迭代。返回迭代历史 list[dict]。"""
    drv = GetDPMotorDriver()
    ia = JS_PK
    # δ=90° 电角度: 与 torque_delta_sweep 逐位一致 ia=sinδ, ib=sin(δ−120°), ic=sin(δ+120°)
    dr = math.radians(90.0)
    ia = JS_PK * math.sin(dr)
    ib = JS_PK * math.sin(dr - 2 * math.pi / 3)     # = −0.5·js (和恒为零, 无零序)
    ic = JS_PK * math.sin(dr + 2 * math.pi / 3)     # = −0.5·js
    t_m, t_cu = 40.0, 40.0                          # 初值 = 环境
    hist = []
    for it in range(1, max_iter + 1):
        hc = HC0 * (1.0 + ABR * (t_m - 20.0))
        out = f"b_map_{tag}_{it}.pos"
        drv.solve(ia=ia, ib=ib, ic=ic, out_bmap=out, hcmag=hc)
        bg1, pole = _gap_bg1(drv, out)
        t_dl = drv.torque(out)
        lptn = thermal_lptn(h_conv=h_conv, bmap=out, verbose=False, t_cu=t_cu)
        tavg = 1.5 * P["p"] * LAM1 * js_to_i(JS_PK)
        row = dict(iter=it, Hc_Am=round(hc, 1), T_m_in=round(t_m, 2),
                   T_cu_in=round(t_cu, 2), Bg1_T=round(bg1, 4),
                   pole_mean_T=round(pole, 4), T_dl_Nm=round(t_dl, 4),
                   T_avg_Nm=round(tavg, 4), P_cu_W=lptn["P_cu_W"],
                   P_fe_W=lptn["P_fe_W"], R_ph_ohm=lptn["R_ph_ohm"],
                   T_winding=lptn["T_winding"], T_magnet=lptn["T_magnet"],
                   T_max=lptn["T_max"], B_max_stator_T=lptn["B_max_stator_T"])
        hist.append(row)
        print(f"[iter {it}] Hc={hc:.3e} (T_m={t_m:.1f}°C) → Bg1={bg1:.4f} T, "
              f"T_dl={t_dl:.4f} Nm, P_cu={lptn['P_cu_W']:.2f} W, "
              f"P_fe={lptn['P_fe_W']:.2f} W → T_w={lptn['T_winding']:.2f}°C, "
              f"T_m={lptn['T_magnet']:.2f}°C")
        if (abs(lptn["T_magnet"] - t_m) < tol
                and abs(lptn["T_winding"] - t_cu) < tol):
            print(f"  收敛: |ΔT_m|={abs(lptn['T_magnet']-t_m):.3f} K, "
                  f"|ΔT_w|={abs(lptn['T_winding']-t_cu):.3f} K (< {tol} K)")
            break
        t_m, t_cu = lptn["T_magnet"], lptn["T_winding"]
    return hist


def efficiency_map(h_conv=15.0, t_cu=None, js_list=(1e6, 2e6, 3e6, 4e6),
                   tag="eff"):
    """效率 Map: 每电流点 1 次 GetDP (取定子 B → 铁耗) + 公式法 T_avg。"""
    drv = GetDPMotorDriver()
    omega = 2 * math.pi * SPEED_RPM / 60.0
    rows = []
    # 铜温: 未给则先跑一次额定 LPTN 取收敛绕组温度
    if t_cu is None:
        t_cu = coupled(h_conv=h_conv)[-1]["T_winding"]
    for js in js_list:
        ia = js
        dr = math.radians(90.0)
        ia = js * math.sin(dr)
        ib = js * math.sin(dr - 2 * math.pi / 3)
        ic = js * math.sin(dr + 2 * math.pi / 3)
        out = f"b_map_{tag}_{js:.0e}.pos"
        drv.solve(ia=ia, ib=ib, ic=ic, out_bmap=out)
        i_pk = js_to_i(js)
        t_avg = 1.5 * P["p"] * LAM1 * i_pk               # δ=90°, sin=1
        p_out = t_avg * omega
        lptn = thermal_lptn(h_conv=h_conv, bmap=out, verbose=False,
                            t_cu=t_cu, js_peak=js)
        p_cu, p_fe = lptn["P_cu_W"], lptn["P_fe_W"]
        eta = p_out / (p_out + p_cu + p_fe) if p_out > 0 else 0.0
        rows.append(dict(js_peak_Am2=js, i_pk_A=round(i_pk, 4),
                         T_avg_Nm=round(t_avg, 4), P_out_W=round(p_out, 2),
                         P_cu_W=round(p_cu, 3), P_fe_W=round(p_fe, 3),
                         eta_pct=round(100 * eta, 2),
                         B_max_stator_T=lptn["B_max_stator_T"],
                         R_ph_ohm=lptn["R_ph_ohm"], T_cu_C=round(t_cu, 1)))
        print(f"[eff js={js:.1e}] i_pk={i_pk:.3f} A → T_avg={t_avg:.3f} Nm, "
              f"P_out={p_out:.1f} W, P_cu={p_cu:.2f} W, P_fe={p_fe:.2f} W, "
              f"η={100*eta:.1f}%")
    return rows


def main():
    print(f"===== 电磁-热耦合迭代 (额定 js={JS_PK:.0e} A/m², δ=90°, "
          f"{SPEED_RPM:.0f} rpm, α_Br={ABR*100:.2f}%/K) =====")
    hist = coupled()
    last = hist[-1]
    print(f"\n===== 效率 Map (T_cu={last['T_winding']:.1f}°C 反馈) =====")
    eff = efficiency_map(t_cu=last["T_winding"])
    rep = dict(coupled_history=hist, coupled_final=last,
               efficiency_map=eff,
               note="耦合: Hc(T_m) GetDP 反馈 + R_ph(T_w) LPTN 反馈; "
                    "效率: 公式法 T_avg + Steinmetz 铁耗, 机械损未计")
    with open(os.path.join(HERE, "coupled_report.json"), "w",
              encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    print("\n→ coupled_report.json")
    return rep


if __name__ == "__main__":
    main()
