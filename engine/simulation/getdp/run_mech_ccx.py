#!/usr/bin/env python3
"""M93 机械应力/过盈压装链 — CalculiX 静力四步 → mech_report.json

链路: --params(设计键, 同 motor_design.json 白名单) → generate_geo.MechStructMesh
      (转子铁芯环+磁钢环 C3D8) → ccx *Static 四步 (额定离心/1.2倍超速/过盈压装
      等效面压/额定磁拉力) → 解析 .dat 积分点应力 → von Mises 峰值+位置/
      界面压力/最小安全系数 → mech_report.json + _progress.json 直通。

Lamé 精度门 (验收): 独立厚壁圆筒验证模型 (同网格机制, 单环) —
  Step1 内压 vs Lamé 厚壁圆筒解析, Step2 离心 vs 旋转圆环解析, 误差 ≤5% 披露。

口径: mech_spec.py 唯一事实源; 过盈压装按解析界面压 p_fit 面压等效施加
  (安装完成态, 非接触算法 — 已披露); 磁拉力按 Maxwell 面压 B²/2μ0 径向 Cload。

用法: python3 run_mech_ccx.py --params ui_params.json --workdir <dir>
      (无 --params 用 constants/motor_design.json 默认设计; workdir 隔离批跑)
"""
import json
import math
import os
import re
import shutil
import subprocess
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from generate_geo import MechStructMesh       # noqa: E402
from load_constants import merged_design      # noqa: E402
from mech_spec import (load_mech_struct, lame_cylinder, rotating_ring,  # noqa: E402
                       interference_pressure, von_mises6, hoop_from_cart,
                       magnetic_pull_pressure)


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


def nset_lines(ids):
    """CalculiX: 每行 ≤16 项; 仅续行以逗号结尾。"""
    ids = [str(i) for i in ids]
    lines = [','.join(ids[k:k + 16]) for k in range(0, len(ids), 16)]
    return [ln + ',' for ln in lines[:-1]] + [lines[-1]]


def _common_blocks(mm, cfg):
    """节点 + 单元 + 材料 + 约束节点选取公共段。"""
    nodes, elems, ang, rr, zs = mm.build()
    by_band = {}
    for e in elems:
        by_band.setdefault(e[2], []).append(e)
    w = []
    A = w.append
    A('*Heading')
    A('M93 mechanical static (auto: run_mech_ccx.py)')
    A('*Node')
    for nid in sorted(nodes):
        x, y, z = nodes[nid]
        A(f'{nid}, {x:.9g}, {y:.9g}, {z:.9g}')
    for band in by_band:
        elset = 'E' + band.upper()
        mat = 'MIRON' if band == 'iron' else 'MMAG'
        iron = band == 'iron'
        A(f'*Element, type=C3D8, Elset={elset}')
        for eid, n, *_ in by_band[band]:
            A(f'{eid}, ' + ', '.join(str(i) for i in n))
    A('*Elset, Elset=ESOLID')
    A(', '.join('E' + b.upper() for b in by_band))
    for band in by_band:
        elset = 'E' + band.upper()
        mat = 'MIRON' if band == 'iron' else 'MMAG'
        iron = band == 'iron'
        A(f'*Solid Section, Elset={elset}, Material={mat}')
        A(f'*Material, Name={mat}')
        A('*Elastic')
        e = cfg['mech_E_iron_pa'] if iron else cfg['mech_E_magnet_pa']
        nu = cfg['mech_nu_iron'] if iron else cfg['mech_nu_magnet']
        A(f'{e:.6g}, {nu:.6g}')
        rho = cfg['mech_rho_iron_kgm3'] if iron else cfg['mech_rho_magnet_kgm3']
        A('*Density')
        A(f'{rho:.6g}')
        if not iron:
            A('*Expansion')
            A(f"{cfg['mech_alpha_magnet_perk']:.6g}")
    # 约束节点: θ≈0 取切向 UY, θ≈90° 取切向 UX (不阻碍轴对称径向膨胀)
    na = len(ang)
    def nearest_node(theta_deg):
        def dang(nid):
            t = np.mod(np.degrees(math.atan2(nodes[nid][1], nodes[nid][0])),
                       360.0)
            return min(abs(t - theta_deg), 360.0 - abs(t - theta_deg))
        return min((mm.nid(0, ia, 0, na) for ia in range(na)), key=dang)
    ta, tb = nearest_node(0.0), nearest_node(90.0)
    z0_nodes = sorted(nid for nid, (x, y, z) in nodes.items() if abs(z) < 1e-12)
    A('*Nset, nset=TANG0')
    A(f'{ta}')
    A('*Nset, nset=TANG90')
    A(f'{tb}')
    A('*Nset, nset=Z0PLANE')
    w.extend(nset_lines(z0_nodes))
    return w, nodes, elems, by_band, (ta, tb)


def magnetic_pull_nodal(mm, by_band, cfg):
    """磁钢最外层 face2 (外圆柱面) Maxwell 面压 → 径向等效节点力。"""
    sigma = magnetic_pull_pressure(cfg['mech_magnetic_pull_b_tesla'])
    nlay = int(cfg['mech_radial_layers_per_band'])
    nodes = mm.nodes
    acc = {}
    for eid, n, band, k, *_ in by_band['magnet']:
        if k != nlay - 1:
            continue
        q = [np.array(nodes[i]) for i in n[4:8]]
        area = 0.5 * float(np.linalg.norm(np.cross(q[2] - q[0], q[3] - q[1])))
        th_mid = math.atan2(sum(p[1] for p in q) / 4.0,
                            sum(p[0] for p in q) / 4.0)
        fx = sigma * area / 4.0 * math.cos(th_mid)
        fy = sigma * area / 4.0 * math.sin(th_mid)
        for i in n[4:8]:
            px, py = acc.get(i, (0.0, 0.0))
            acc[i] = (px + fx, py + fy)
    return [(nid, fx, fy) for nid, (fx, fy) in sorted(acc.items())], sigma


def build_rotor_inp(mm, cfg, design, dT, f_pull):
    """装配转子 4 步静力 .inp 内容行 (过盈用热缩等效 + 最小约束)。"""
    w, nodes, elems, by_band, (ta, tb) = _common_blocks(mm, cfg)
    A = w.append
    mag_nodes = sorted({i for e in by_band['magnet'] for i in e[1]})
    A('*Nset, nset=NMAG')
    w.extend(nset_lines(mag_nodes))
    A('*Nset, nset=ALLNODES')
    w.extend(nset_lines(sorted(nodes)))
    A('*Initial Conditions, type=TEMPERATURE')
    A('ALLNODES, 0.0')
    w2_rated = (2 * math.pi * cfg['mech_rated_rpm'] / 60.0) ** 2
    w2_over = (cfg['mech_overspeed_factor'] * 2 * math.pi
               * cfg['mech_rated_rpm'] / 60.0) ** 2
    centrif = lambda w2: [f'EIRON, CENTRIF, {w2:.6E}, 0., 0., 0., 0., 0., 1.',
                          f'EMAGNET, CENTRIF, {w2:.6E}, 0., 0., 0., 0., 0., 1.']
    steps = [
        ('Step1: 额定转速离心', ('Dload', centrif(w2_rated))),
        ('Step2: 1.2 倍超速离心', ('Dload', centrif(w2_over))),
        ('Step3: 磁钢过盈压装预应力 (热缩等效, ΔT 施于 NMAG)',
         ('Temperature', [('NMAG', dT)])),
        ('Step4: 额定磁拉力 (Maxwell 面压径向节点力)',
         ('Cload', [(nid, 1, fx) for nid, fx, _ in f_pull]
          + [(nid, 2, fy) for nid, _, fy in f_pull])),
    ]
    first = True
    for title, (kind, items) in steps:
        A(f'** ---- {title} ----')
        A('*Step')
        A('*Static')
        if first:
            A('** 最小约束防刚体位移: 切向 UX/UY 单点 + z=0 面 UZ')
            A('*Boundary')
            A('TANG0, 2, 2, 0.0')
            A('TANG90, 1, 1, 0.0')
            A('Z0PLANE, 3, 3, 0.0')
            first = False
        A(f'*{kind}, OP=NEW')
        for it in items:
            if kind == 'Cload':
                A(f'{it[0]}, {it[1]}, {it[2]:.6E}')
            elif kind == 'Temperature':
                A(f'{it[0]}, {it[1]:.6E}')
            else:
                A(it)
        A('*El Print, Elset=ESOLID')
        A('S')
        A('*End Step')
    return w


def build_lame_inp(mm, cfg, design, p_in):
    """Lamé 验证 .inp: 单环厚壁圆筒 — Step1 内压, Step2 离心。"""
    w, nodes, elems, by_band, (ta, tb) = _common_blocks(mm, cfg)
    A = w.append
    w2 = (2 * math.pi * cfg['mech_rated_rpm'] / 60.0) ** 2
    A('** ---- Step1: 内压厚壁圆筒 (Lamé 对照) ----')
    A('*Step')
    A('*Static')
    A('*Boundary')
    A('TANG0, 2, 2, 0.0')
    A('TANG90, 1, 1, 0.0')
    A('Z0PLANE, 3, 3, 0.0')
    A('*Elset, Elset=EIRON0')
    w.extend(nset_lines([e[0] for e in by_band['iron'] if e[3] == 0]))
    A('*Dload, OP=NEW')
    A(f'EIRON0, P1, {p_in:.6E}')
    A('*El Print, Elset=ESOLID')
    A('S')
    A('*End Step')
    A('** ---- Step2: 旋转圆环 (解析对照) ----')
    A('*Step')
    A('*Static')
    A('*Dload, OP=NEW')
    A(f'EIRON, CENTRIF, {w2:.6E}, 0., 0., 0., 0., 0., 1.')
    A('*El Print, Elset=ESOLID')
    A('S')
    A('*End Step')
    return w


def parse_dat(path):
    """解析 .dat 应力块 → [step][ (eid, ip, (sxx,syy,szz,sxy,sxz,syz)) ]。"""
    steps = []
    cur = None
    in_block = False
    with open(path, encoding='utf-8', errors='replace') as f:
        for ln in f:
            if ln.startswith(' stresses'):
                cur = []
                steps.append(cur)
                in_block = True
                continue
            if not in_block:
                continue
            t = ln.split()
            if len(t) == 8 and re.fullmatch(r'\d+', t[0]):
                cur.append((int(t[0]), int(t[1]),
                            tuple(float(v) for v in t[2:8])))
            elif t and not re.fullmatch(r'[-+\d.eE]+', t[0]):
                in_block = False
    return steps


def step_stats(blocks, mm):
    """一步应力块 → (vM 峰值元组, {(band, ir_band): hoop 均值})。

    vM 峰值元组: (vm, band, ir_band, r_centroid, theta_deg, z_centroid)。
    """
    eid_info = {e[0]: e for e in mm.elems}
    peak = None
    hoop_sum, hoop_n = {}, {}
    for eid, ip, s in blocks:
        vm = von_mises6(*s)
        e = eid_info[eid]
        if peak is None or vm > peak[0]:
            cx = sum(mm.nodes[i][0] for i in e[1]) / 8.0
            cy = sum(mm.nodes[i][1] for i in e[1]) / 8.0
            cz = sum(mm.nodes[i][2] for i in e[1]) / 8.0
            peak = (vm, e[2], e[3], e[4],
                    float(np.degrees(math.atan2(cy, cx)) % 360.0), float(cz))
        nid0 = e[1][0]
        th = math.atan2(mm.nodes[nid0][1], mm.nodes[nid0][0])
        key = (e[2], e[3])
        hoop_sum[key] = (hoop_sum.get(key, 0.0)
                         + hoop_from_cart(s[0], s[1], s[3], th))
        hoop_n[key] = hoop_n.get(key, 0) + 1
    return peak, {k: hoop_sum[k] / hoop_n[k] for k in hoop_sum}


def run(workdir, params=None):
    os.makedirs(workdir, exist_ok=True)
    cfg = load_mech_struct()
    design = merged_design(params)
    progress(workdir, 'mesh', 10, None)

    mm = MechStructMesh(design, cfg)
    mm.build()
    na = len(mm.ang)
    by_band = {}
    for e in mm.elems:
        by_band.setdefault(e[2], []).append(e)
    mesh_info = {'n_nodes': len(mm.nodes), 'n_elem': len(mm.elems),
                 'angles': na, 'radial_layers': mm.nr,
                 'axial_layers': int(cfg['mech_axial_layers'])}

    a, ai, bo = design['R_ri'], design['R_shaft'], mm.R_rm
    p_fit = interference_pressure(cfg['mech_interference_radial_um'] * 1e-6,
                                  a, ai, bo,
                                  cfg['mech_E_iron_pa'], cfg['mech_nu_iron'],
                                  cfg['mech_E_magnet_pa'],
                                  cfg['mech_nu_magnet'])
    # 过盈热缩等效: 磁钢自由收缩 δ → ΔT = δ/(α·R_ri), 绑定界面强迫回弹产生压装残余应力
    dT = -(cfg['mech_interference_radial_um'] * 1e-6
           / (cfg['mech_alpha_magnet_perk'] * a))
    f_pull, sigma_pull = magnetic_pull_nodal(mm, by_band, cfg)

    with open(os.path.join(workdir, 'rotor_mech.inp'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(build_rotor_inp(mm, cfg, design, dT, f_pull))
                + '\n')
    progress(workdir, 'solve', 30, None)

    ccx = resolve_ccx()

    def ccx_run(job):
        r = subprocess.run([ccx, '-i', job, '-nt', '2'], cwd=workdir,
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f'ccx 失败 exit={r.returncode}: '
                               f'{(r.stderr or r.stdout)[-400:]}')
    ccx_run(os.path.join(workdir, 'rotor_mech'))
    progress(workdir, 'solve', 60, None)

    # Lamé 验证模型 (单环铁芯 0.02..0.05 m)
    lame_mm = MechStructMesh(design, cfg, bands=[('iron', 0.02, 0.05)])
    p_in = 5.0e6
    with open(os.path.join(workdir, 'lame_check.inp'), 'w',
              encoding='utf-8') as f:
        f.write('\n'.join(build_lame_inp(lame_mm, cfg, design, p_in)) + '\n')
    ccx_run(os.path.join(workdir, 'lame_check'))
    progress(workdir, 'post', 80, None)

    yield_of = {'iron': cfg['mech_yield_iron_pa'],
                'magnet': cfg['mech_yield_magnet_pa']}
    blocks = parse_dat(os.path.join(workdir, 'rotor_mech.dat'))
    if len(blocks) != 4:
        raise RuntimeError(f'rotor_mech.dat 应力块数 {len(blocks)} != 4')
    names = ['centrifugal_rated', 'centrifugal_overspeed',
             'interference_fit', 'magnetic_pull']
    scenarios = {}
    hoop_by_step = []
    for name, blk in zip(names, blocks):
        peak, hoop = step_stats(blk, mm)
        vm, band = peak[0], peak[1]
        scenarios[name] = {
            'vm_peak_pa': vm, 'vm_peak_mpa': vm / 1e6,
            'loc': {'band': band, 'r_mm': peak[3] * 1e3,
                    'theta_deg': peak[4], 'z_mm': peak[5]},
            'min_sf': yield_of[band] / vm,
            'yield_pa': yield_of[band]}
        hoop_by_step.append(hoop)

    lblocks = parse_dat(os.path.join(workdir, 'lame_check.dat'))
    if len(lblocks) != 2:
        raise RuntimeError(f'lame_check.dat 应力块数 {len(lblocks)} != 2')
    a0, b0, nlay = 0.02, 0.05, int(cfg['mech_radial_layers_per_band'])
    g = 1.0 / math.sqrt(3.0)
    gp_frac = ((1.0 - g) / 2.0, (1.0 + g) / 2.0)

    def layer_gp_radii(r_lo, r_hi):
        """该径向层 2×2×2 积分点的半径 (只依赖 ζ=±1/√3)。"""
        return [r_lo + f * (r_hi - r_lo) for f in gp_frac]

    def ana_mean(fn, layer):
        """解析函数在层积分点半径上的等权均值 (与 FEM El Print 采样同权)。"""
        r_lo, r_hi = a0 + layer * (b0 - a0) / nlay, a0 + (layer + 1) * (b0 - a0) / nlay
        return float(np.mean([fn(r) for r in layer_gp_radii(r_lo, r_hi)]))

    _, h_p = step_stats(lblocks[0], lame_mm)
    _, h_c = step_stats(lblocks[1], lame_mm)

    def gate(fem, ana):
        err = abs(fem - ana) / abs(ana) * 100.0
        return {'fem_mpa': fem / 1e6, 'analytic_mpa': ana / 1e6,
                'err_pct': err, 'pass': bool(err <= cfg['mech_lame_tol_pct'])}
    lame_check = {'r_eval_mm': [round(r * 1e3, 3)
                                for r in layer_gp_radii(a0, b0)],
                  'internal_pressure': gate(
                      h_p[('iron', 0)],
                      ana_mean(lambda r: lame_cylinder(p_in, 0.0, a0, b0, r)[1],
                               0)),
                  'centrifugal': gate(
                      h_c[('iron', 0)],
                      ana_mean(lambda r: rotating_ring(
                          cfg['mech_rho_iron_kgm3'],
                          2 * math.pi * cfg['mech_rated_rpm'] / 60.0,
                          cfg['mech_nu_iron'], a0, b0, r)[1], 0)),
                  'tol_pct': cfg['mech_lame_tol_pct']}

    # 过盈步披露: 磁钢内层/铁芯外层 hoop FEM vs 过盈 Lamé 自由环解析 (热缩等效)
    def band_ana(fn, band_r, layer):
        r_lo, r_hi = band_r
        rlo = r_lo + layer * (r_hi - r_lo) / nlay
        rhi = r_lo + (layer + 1) * (r_hi - r_lo) / nlay
        return float(np.mean([fn(r) for r in layer_gp_radii(rlo, rhi)]))

    mag_hoop_fem = hoop_by_step[2].get(('magnet', 0))
    iron_hoop_fem = hoop_by_step[2].get(('iron', nlay - 1))
    ana_mag_hoop = band_ana(lambda r: lame_cylinder(p_fit, 0.0, a, bo, r)[1],
                            (a, bo), 0)
    ana_iron_hoop = band_ana(lambda r: lame_cylinder(0.0, p_fit, ai, a, r)[1],
                             (ai, a), nlay - 1)

    report = {
        'meta': {'ccx': ccx,
                 'params': {k: design[k] for k in ('R_shaft', 'R_ri', 'mag_t',
                                                   'n_poles', 'mag_half_deg',
                                                   'L_stack')},
                 'mesh': mesh_info,
                 'boundary': '最小约束: θ≈0 切向 UY + θ≈90° 切向 UX + z=0 面 UZ'},
        'interface_fit': {
            'delta_r_um': cfg['mech_interference_radial_um'],
            'method': '热缩等效: 磁钢 ΔT=−δ/(α·R_ri), 绑定界面回弹产生压装残余应力',
            'alpha_magnet_perk': cfg['mech_alpha_magnet_perk'],
            'delta_t_k': dT,
            'p_fit_pa': p_fit, 'p_fit_mpa': p_fit / 1e6,
            'fem_magnet_inner_hoop_mpa': (mag_hoop_fem / 1e6
                                          if mag_hoop_fem is not None
                                          else None),
            'lame_magnet_hoop_mpa': ana_mag_hoop / 1e6,
            'fem_iron_outer_hoop_mpa': (iron_hoop_fem / 1e6
                                        if iron_hoop_fem is not None
                                        else None),
            'lame_iron_hoop_mpa': ana_iron_hoop / 1e6,
            'note': '披露非门: 磁钢内层 hoop 应为拉 (Lamé 内压), 铁芯外层应为压 '
                    '(Lamé 外压); FEM 与解析偏差来自极弧非整圆 + 离散。'
                    '注意: 热缩等效下磁钢 hoop 为拉应力, NdFeB 抗拉强度 '
                    '(~80MPa) 远低于抗压判据 mech_yield_magnet_pa, '
                    '过盈量加大有磁体开裂风险, 需结合抗拉口径评估'},
        'magnetic_pull': {'b_tesla': cfg['mech_magnetic_pull_b_tesla'],
                          'sigma_gap_pa': sigma_pull,
                          'note': 'σ = B²/2μ0 施于转子外表面 (径向拉)'},
        'scenarios': scenarios,
        'lame_check': lame_check,
        'check_pass': bool(lame_check['internal_pressure']['pass']
                           and lame_check['centrifugal']['pass']),
    }
    with open(os.path.join(workdir, 'mech_report.json'), 'w',
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
    print(f"mech_report.json 已生成: check_pass={rep['check_pass']}")
    for name, sc in rep['scenarios'].items():
        print(f"  {name}: vM={sc['vm_peak_mpa']:.2f} MPa "
              f"@{sc['loc']['band']} r={sc['loc']['r_mm']:.2f}mm "
              f"θ={sc['loc']['theta_deg']:.0f}° SF={sc['min_sf']:.2f}")
    print('界面压装:', rep['interface_fit'])
    print('Lamé 门:', json.dumps(rep['lame_check'], ensure_ascii=False))
