#!/usr/bin/env python3
"""M90 定子结构模态链 — CalculiX 自由+约束模态 → modal_report.json

链路: --params(设计键, 同 motor_design.json 白名单) → generate_geo.StatorStructMesh
      (定子实体 C3D8, 叠片 ANISO 等效口径 modal_spec) → ccx(*Frequency 两步:
      Step1 自由(多求6阶覆盖刚体) / Step2 外圆 OUTER 固支) → 解析 .dat 特征值
      + 振型(中轭圆径向位移节径投影) → modal_report.json + _progress.json 直通。

用法: python3 run_modal_ccx.py --params ui_params.json --workdir <dir>
      (无 --params 时用 constants/motor_design.json 默认设计; workdir 隔离批跑)
"""
import json
import os
import re
import shutil
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from generate_geo import StatorStructMesh          # noqa: E402
from load_constants import merged_design           # noqa: E402
from modal_spec import (load_nvh_struct, ring_mode_hz,   # noqa: E402
                        classify_nodal_diameter, mode_label)


def progress(workdir, stage, percent, eta_s=None):
    with open(os.path.join(workdir, '_progress.json'), 'w', encoding='utf-8') as f:
        json.dump({'stage': stage, 'percent': percent, 'eta_s': eta_s}, f)


def resolve_ccx():
    """优先 bin/third_party 实测版本 ccx_2.22, 兜底 PATH 内 ccx。"""
    root = os.path.abspath(os.path.join(_HERE, '..', '..', '..', '..', '..'))
    cand = os.path.join(root, 'bin', 'third_party', 'calculix', 'bin', 'ccx_2.22')
    if os.access(cand, os.X_OK):
        return cand
    which = shutil.which('ccx')
    if which:
        return which
    raise FileNotFoundError('未找到 CalculiX (bin/third_party/calculix 或 PATH)')


def fnum(s):
    """Fortran 掉 E 浮点修复: '8.455148-100' → 8.455148e-100。"""
    return float(re.sub(r'(\d)([+-]\d{2,})$', r'\1e\2', s))


def parse_dat(path):
    """解析 ccx .dat → (steps, shapes)。

    steps: [{mode, eigenvalue, f_hz}];  shapes: {(step, mode): {nid: (ux,uy,uz)}}
    """
    steps, shapes = [], {}
    step_idx = -1
    cur_mode = None
    in_eig_table = False
    in_disp = False
    with open(path, encoding='utf-8', errors='replace') as f:
        for ln in f:
            if 'S T E P' in ln:
                step_idx += 1
                steps.append([])
                in_eig_table = in_disp = False
            elif 'E I G E N V A L U E   O U T P U T' in ln:
                in_eig_table = True
                in_disp = False
                continue
            elif ('P A R T I C I P A T I O N' in ln or
                  'E F F E C T I V E   M A S S' in ln or
                  ln.strip().startswith('TOTAL')):
                in_eig_table = False
                in_disp = False
                continue
            elif 'E I G E N V A L U E    N U M B E R' in ln:
                cur_mode = int(ln.split()[-1])
                in_eig_table = False
                continue
            elif ln.startswith(' displacements'):
                in_disp = True
                continue
            if in_eig_table:
                t = ln.split()
                if len(t) == 5 and re.fullmatch(r'\d+', t[0]):
                    steps[step_idx].append({'mode': int(t[0]),
                                            'eigenvalue': float(t[1]),
                                            'f_hz': float(t[3])})
            elif in_disp and cur_mode is not None:
                t = ln.split()
                if len(t) == 4 and re.fullmatch(r'\d+', t[0]):
                    key = (step_idx, cur_mode)
                    shapes.setdefault(key, {})[int(t[0])] = \
                        (fnum(t[1]), fnum(t[2]), fnum(t[3]))
    return steps, shapes


def split_steps(all_rows):
    """按 'S T E P' 已由 parse_dat 计步 → 返回 (free_rows, fix_rows)。"""
    free = [r for r in all_rows if r['mode'] <= 500 and r['eigenvalue'] is not None]
    return free


def _mode_spectrum(sm, shp, cfg):
    """全部轭/齿圈 × 全部轴向层累积径向位移功率谱 P[m]。"""
    na = len(sm.ang)
    r_tooth_hi = sm.R_si + sm.slot_depth
    P = np.zeros(cfg['nvh_struct_m_max'] + 1)
    for ir, r in enumerate(sm.rr):
        if r < r_tooth_hi - 1e-12:
            continue
        for iz in range(len(sm.zs)):
            th, ur = [], []
            for ia, t in enumerate(sm.ang):
                nid = sm.nid(ir, ia, iz, na)
                if nid not in shp:
                    continue
                ux, uy, _ = shp[nid]
                th.append(np.deg2rad(t))
                ur.append(ux * np.cos(np.deg2rad(t)) + uy * np.sin(np.deg2rad(t)))
            if len(ur) < 8:
                continue
            th = np.asarray(th)
            ur = np.asarray(ur)
            ur = ur / (np.linalg.norm(ur) + 1e-30)
            c = ur @ np.exp(-1j * np.outer(th, np.arange(len(P))))
            P += np.abs(c) ** 2
    return P


def classify_modes(sm, shapes, step_idx, rows, cfg):
    """径向位移角向功率谱 argmax → 节径数 m + 类别标签。"""
    out = []
    for r in rows:
        shp = shapes.get((step_idx, r['mode']), {})
        if not shp:
            m, amp, lb = -1, 0.0, 'unresolved(未解析)'
        else:
            P = _mode_spectrum(sm, shp, cfg)
            m = int(np.argmax(P))
            amp = float(P[m] / (P.sum() + 1e-30))
            lb = mode_label(m, f_hz=r['f_hz'],
                            f_rigid_hz=cfg['nvh_struct_f_rigid_hz'])
        out.append({'mode': r['mode'], 'f_hz': r['f_hz'],
                    'nodal_diameter': m, 'shape_amp': round(amp, 3),
                    'label': lb})
    return out


def run(workdir, params=None):
    os.makedirs(workdir, exist_ok=True)
    cfg = load_nvh_struct()
    design = merged_design(params)
    progress(workdir, 'mesh', 10, None)

    sm = StatorStructMesh(design, cfg)
    inp = os.path.join(workdir, 'motor_stator_modal.inp')
    info = sm.write_inp(inp)
    progress(workdir, 'solve', 25, None)

    ccx = resolve_ccx()
    job = os.path.join(workdir, 'motor_stator_modal')
    r = subprocess.run([ccx, '-i', job, '-nt', '2'], cwd=workdir,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'ccx 失败 exit={r.returncode}: '
                           f'{(r.stderr or r.stdout)[-400:]}')
    progress(workdir, 'post', 75, None)

    steps, shapes = parse_dat(job + '.dat')
    # steps[0] = Step1(自由, n_modes_free 行), steps[1] = Step2(约束)
    free_rows = steps[0]
    fix_rows = steps[1] if len(steps) > 1 else []
    if len(free_rows) < info['n_modes_free'] or \
            len(fix_rows) < info['n_modes_constrained']:
        raise RuntimeError(f'.dat 解析不全: free={len(free_rows)} '
                           f'fix={len(fix_rows)}')
    f_rigid = cfg['nvh_struct_f_rigid_hz']
    free_el = [r for r in free_rows if r['f_hz'] > f_rigid][:cfg['nvh_struct_n_modes']]
    free_cls = classify_modes(sm, shapes, 0, free_el, cfg)
    fix_cls = classify_modes(sm, shapes, 1, fix_rows[:cfg['nvh_struct_n_modes']], cfg)

    # 薄环解析对照 (口径: 轭中径/轭厚, 面内等效 E_ip + rho_eff)
    r_tooth_hi = design['R_si'] + design['slot_depth']
    yoke_R = 0.5 * (r_tooth_hi + design['R_so'])
    yoke_t = design['R_so'] - r_tooth_hi
    f_ring2 = float(ring_mode_hz(2, yoke_R, yoke_t, sm.e_ip, sm.rho_eff))
    fem_m2 = next((c['f_hz'] for c in free_cls if c['nodal_diameter'] == 2), None)

    f_free = [c['f_hz'] for c in free_cls]
    f_fix = [c['f_hz'] for c in fix_cls]
    tol = 1e-6 * max(f_free + f_fix) if (f_free and f_fix) else 0.0
    # 退化对 (sin/cos 同频成对) 允许相等 → 非降序判据
    check = {
        'free_ascending': all(b >= a - tol for a, b in zip(f_free, f_free[1:])),
        'fix_ascending': all(b >= a - tol for a, b in zip(f_fix, f_fix[1:])),
        'fix_stiffer_than_free': f_fix and f_free and f_fix[0] > f_free[0] + tol,
        'm2_classified': fem_m2 is not None,
    }
    report = {
        'meta': {'ccx': ccx, 'params': {k: design[k] for k in
                                        ('R_si', 'slot_depth', 'R_so', 'n_slots',
                                         'slot_half_deg', 'L_stack')},
                 'mesh': {k: info[k] for k in ('n_nodes', 'n_elem', 'angles',
                                               'radial_layers', 'axial_layers')},
                 'lamination': {'E_ip_pa': sm.e_ip, 'E_ax_pa': sm.e_ax,
                                'G_pa': sm.g12, 'rho_eff_kgm3': sm.rho_eff,
                                'k_fill': cfg['nvh_struct_k_fill']},
                 'mount': {'n_points': 4, 'angles_deg': [45, 135, 225, 315],
                           'k_n_per_m': cfg['nvh_struct_k_mount_n_per_m'],
                           'note': 'Step1 移除弹簧=自由模态; Step2 挂弹簧=装机口径 '
                                   '(*Model Change)'}},
        'free': free_cls,
        'constrained': fix_cls,
        'ring_check': {'yoke_R_m': yoke_R, 'yoke_t_m': yoke_t,
                       'ring_m2_formula_hz': round(f_ring2, 1),
                       'fem_free_m2_hz': fem_m2,
                       'ratio_fem_over_ring': (round(fem_m2 / f_ring2, 3)
                                               if fem_m2 else None),
                       'note': '薄环公式为量级对照 (齿/槽质量未计入), 非精度门'},
        'check': check,
        'check_pass': all(check.values()),
    }
    with open(os.path.join(workdir, 'modal_report.json'), 'w',
              encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    progress(workdir, 'done', 100, 0)
    return report


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
    print(f"modal_report.json 已生成: check_pass={rep['check_pass']}")
    for tag, rows in (('自由', rep['free']), ('约束', rep['constrained'])):
        print(f'-- {tag}模态 --')
        for c in rows:
            print(f"  {c['mode']:>3} {c['f_hz']:10.1f} Hz  m={c['nodal_diameter']:>2}  "
                  f"{c['label']}")
    print('薄环对照:', rep['ring_check'])
