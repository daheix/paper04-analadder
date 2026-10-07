#!/usr/bin/env python3
"""M91 声辐射解析链 — 力谱×模态叠加×圆柱谐波辐射效率 → acoustic_report.json

链路: force_fft_report.json (M89 order×mode 力谱 F[h][m], Pa)
      + modal_report.json (M90 结构模态表, 节径数 m ↔ 频率)
      + 设计键 (R_so/L_stack) → 逐 (h,m):
      f=h·f_e → 模态速度 v=|F|·|H_v(f;f_m,ζ)| (力谱节径 m 配对结构模态,
      缺失时薄环解析公式兜底) → σ_m(k·a) 圆柱谐波辐射效率 (acoustic_spec)
      → W=½ρc·σ·v²·S → L_w=10log10(W/W_ref) → L_p=L_w−10log10(2πr²)
      → acoustic_report.json + _progress.json 直通。

用法: python3 run_acoustic_radiation.py [--params ui_params.json] [--workdir DIR]
      输入报告查找顺序: workdir/ → workdir/../force_fft|modal/ (四段编排约定)。
口径卡: acoustic_spec.py (doctest 可执行, constants/nvh_acoustic.json 单一事实源)。
"""
import json
import math
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from load_constants import merged_design                    # noqa: E402
from modal_spec import load_nvh_struct                      # noqa: E402
import acoustic_spec as ac                                  # noqa: E402


def progress(workdir, stage, percent, eta_s=None):
    with open(os.path.join(workdir, '_progress.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'stage': stage, 'percent': percent, 'eta_s': eta_s}, f)


def find_input(workdir, names, tag):
    """报告查找: workdir/ → workdir/../<tag>/ (四段编排目录约定), fail-fast。"""
    for d in (workdir, os.path.join(workdir, '..', tag)):
        for n in names:
            p = os.path.join(d, n)
            if os.path.isfile(p):
                return p
    raise FileNotFoundError(
        f"未找到 {tag} 输入报告 {names}: 应位于 workdir 或 workdir/../{tag}/ "
        f"(四段编排: emag_t→force_fft→modal→acoustic)")


def yoke_geometry(design, modal_report):
    """轭几何: 优先 modal_report.ring_check 披露值, 兜底设计键推导 (中径/厚度)。"""
    ring = modal_report.get('ring_check', {})
    r_yoke = ring.get('yoke_R_m')
    t_yoke = ring.get('yoke_t_m')
    if not (r_yoke and t_yoke):
        r_si = float(design['R_si'])
        r_so = float(design['R_so'])
        t_yoke = r_so - r_si
        r_yoke = (r_si + r_so) / 2.0
    return float(r_yoke), float(t_yoke)


def run(workdir, params=None):
    os.makedirs(workdir, exist_ok=True)
    t_start = time.time()
    progress(workdir, 'start', 2, None)
    design = merged_design(params or {})

    progress(workdir, 'load_reports', 8, None)
    fft_path = find_input(workdir, ['force_fft_report.json'], 'force_fft')
    modal_path = find_input(workdir, ['modal_report.json'], 'modal')
    with open(fft_path, encoding='utf-8') as f:
        fft_rep = json.load(f)
    with open(modal_path, encoding='utf-8') as f:
        modal_rep = json.load(f)

    cfg = ac.load_nvh_acoustic()
    struct_cfg = load_nvh_struct()
    zeta = float(cfg['nvh_acoustic_zeta_struct'])
    fe = float(fft_rep['params']['f_el_hz'])
    r_so = float(design['R_so'])
    l_stack = float(design['L_stack'])
    r_yoke, t_yoke = yoke_geometry(design, modal_rep)
    # 兜底薄环材料: 优先 FEM 叠片等效披露值 (modal_report.meta.lamination)
    lam = modal_rep.get('meta', {}).get('lamination', {})
    e_ip = float(lam.get('E_ip_pa', struct_cfg['nvh_struct_E_iron_pa']
                         * struct_cfg['nvh_struct_k_fill']))
    rho_eff = float(lam.get('rho_eff_kgm3',
                            struct_cfg['nvh_struct_rho_iron']
                            * struct_cfg['nvh_struct_k_fill']))

    orders = fft_rep['spectrum']['orders']
    modes = fft_rep['spectrum']['modes']
    F = np.asarray(fft_rep['spectrum']['F_Pa'], dtype=float)

    # 结构模态配对表: FEM 节径频率优先, 缺失节径薄环解析兜底
    fem_modes = [(int(c['nodal_diameter']), float(c['f_hz']))
                 for c in modal_rep.get('free', [])
                 if c.get('f_hz', 0) >= struct_cfg['nvh_struct_f_rigid_hz']]
    n_rad_max = int(cfg['nvh_acoustic_n_rad_max'])
    st_table = ac.match_structural_modes(fem_modes, R=r_yoke, t=t_yoke,
                                         E=e_ip, rho=rho_eff, zeta=zeta)
    st_by_m = {int(row['m']): row for row in st_table}

    # 磁致伸缩常量开关 (默认关): 力谱整体 (1+α)
    ms_gain = ac.magnetostriction_gain(
        bool(cfg['nvh_acoustic_magnetostriction_enabled']),
        float(cfg['nvh_acoustic_magnetostriction_alpha']))
    S = math.pi * 2.0 * r_so * l_stack \
        * float(cfg['nvh_acoustic_radiating_frac'])

    progress(workdir, 'acoustic_solve', 20, None)
    spl_rows = []
    w_total = 0.0
    n_pair, n_skip = 0, 0
    for i, h in enumerate(orders):
        if h <= 0.0:
            continue                       # DC 静力分量无声辐射
        f_hz = h * fe
        ka = 2.0 * math.pi * f_hz * r_so / cfg['nvh_acoustic_c_air_m_s']
        for j, m in enumerate(modes):
            f_amp = float(F[i, j]) * ms_gain
            if f_amp <= 0.0:
                continue
            m_eff = min(int(m), n_rad_max)   # 高节径按 n_rad_max 口径 (辐射更弱, 保守)
            st = st_by_m.get(m_eff)
            if st is None or st['f_hz'] is None:
                n_skip += 1
                continue
            # 模态力口径: 面压 p·S (振型平均压力×辐射面积, 单位模态质量)
            v = abs(f_amp) * S * abs(ac.modal_frf(f_hz, st['f_hz'], zeta))
            sigma = ac.radiation_efficiency(m_eff, ka)
            w = ac.radiated_power(v, sigma, S)
            w_total += w
            n_pair += 1
            spl_rows.append({
                'h': float(h), 'm': int(m),
                'f_hz': round(f_hz, 2), 'force_Pa': round(f_amp, 3),
                'f_struct_hz': round(st['f_hz'], 1),
                'struct_source': st['source'],
                'sigma_n': round(sigma, 6),
                'v_m_s': round(v, 9),
                'spl_db': round(ac.spl_at_1m(ac.swl_db(max(w, 1e-30))), 2),
            })
        progress(workdir, 'acoustic_solve',
                 20 + 70 * (i + 1) / len(orders), None)

    spl_rows.sort(key=lambda r: -r['spl_db'])
    dom = spl_rows[:int(fft_rep['params'].get('top_n', 10))]
    lw_total = ac.swl_db(max(w_total, 1e-30))
    lp_total = ac.spl_at_1m(lw_total)

    fallback_n = sum(1 for r in st_table if r['fallback'])
    check = {
        'rows_nonempty': len(spl_rows) > 0,
        'sigma_finite': all(math.isfinite(r['sigma_n'])
                            and r['sigma_n'] >= 0 for r in spl_rows),
        'spl_descending': all(a['spl_db'] >= b['spl_db'] - 1e-9
                              for a, b in zip(dom, dom[1:])),
        'pairs_matched': n_pair > 0,
    }
    rep = {
        'params': {
            'force_fft_report': os.path.abspath(fft_path),
            'modal_report': os.path.abspath(modal_path),
            'f_el_hz': fe, 'R_so_m': r_so, 'L_stack_m': l_stack,
            'yoke_R_m': round(r_yoke, 5), 'yoke_t_m': round(t_yoke, 5),
            'radiating_area_m2': round(S, 5), 'zeta_struct': zeta,
            'n_rad_max': n_rad_max,
            'magnetostriction': {
                'enabled': bool(cfg['nvh_acoustic_magnetostriction_enabled']),
                'alpha': float(cfg['nvh_acoustic_magnetostriction_alpha']),
                'gain': ms_gain,
            },
            'rho_air_kgm3': cfg['nvh_acoustic_rho_air_kgm3'],
            'c_air_m_s': cfg['nvh_acoustic_c_air_m_s'],
            'r_meas_m': cfg['nvh_acoustic_r_meas_m'],
            'sigma_formula': '2/(pi*ka*eps_n*|H_n(ka)|^2)*Re(H_n\'/(i*H_n))',
            'lp_formula': 'L_p = L_w - 10*log10(2*pi*r^2) (半球自由场)',
        },
        'structural_table': st_table,
        'spl_table': spl_rows,
        'dominant_spl': dom,
        'check': dict(check, structural_fallback_count=fallback_n,
                      pairs_matched=n_pair, pairs_skipped=n_skip,
                      lw_total_db=round(lw_total, 2),
                      lp_1m_db=round(lp_total, 2)),
        'runtime_s': round(time.time() - t_start, 2),
        'log': [f'输入: {os.path.abspath(fft_path)}',
                f'输入: {os.path.abspath(modal_path)}',
                f'配对 {n_pair} 项 / 跳过 {n_skip} 项 / '
                f'薄环兜底节径 {fallback_n} 项',
                f'L_w={lw_total:.2f} dB, L_p@1m={lp_total:.2f} dB'],
        'check_pass': all(check.values()),
    }
    with open(os.path.join(workdir, 'acoustic_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    progress(workdir, 'done', 100, 0)
    return rep


if __name__ == '__main__':
    wd = os.path.dirname(os.path.abspath(__file__))
    params = None
    if '--params' in sys.argv:
        pf = sys.argv[sys.argv.index('--params') + 1]
        params = json.load(open(pf, encoding='utf-8'))
        params = {k: v for k, v in params.items() if not k.startswith('_')}
        wd = os.path.dirname(os.path.abspath(pf))
    if '--workdir' in sys.argv:
        wd = os.path.abspath(sys.argv[sys.argv.index('--workdir') + 1])
    rep = run(wd, params)
    print(f"acoustic_report.json 已生成: check_pass={rep['check_pass']}")
    for r in rep['dominant_spl'][:5]:
        print(f"  h={r['h']:.0f} m={r['m']:>2} f={r['f_hz']:8.1f} Hz "
              f"σ={r['sigma_n']:.4f} SPL={r['spl_db']:6.2f} dB ({r['struct_source']})")
    print('L_w 总声功率级 %.2f dB, L_p@1m %.2f dB'
          % (rep['check']['lw_total_db'], rep['check']['lp_1m_db']))
