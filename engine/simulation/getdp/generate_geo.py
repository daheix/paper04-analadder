#!/usr/bin/env python3
"""
4极12槽表贴式永磁电机(PM-SPM) 2D截面 Gmsh .geo 生成器

结构化扇区网格思路（对标商业工具 RMxprt/Radpro 几何规范）：
- 角度网格线 = 全部槽边缘角 + 磁体边缘角（可再细分加密）
- 半径网格线 = 轴/转子铁/磁体/气隙/齿槽/定子轭各圆
- 每个网格单元 = 一个环带 × 一个角扇区 → 按 (r,θ) 判定材料并归入物理组

物理组（Physical Surface，2D）:
  shaft, rotor_iron, magnet_N, magnet_S, airgap, stator_iron,
  SlotAp/SlotAn/SlotBp/SlotBn/SlotCp/SlotCn（6组×2槽，双层整距绕组净槽电流分布）
物理曲线: Outer (定子外圆, Dirichlet Az=0)

单位: SI (m, T, A/m)
"""
import numpy as np
import json
import os


class SectorGeoWriter:
    """通用参数化几何生成器 — 单一事实源 constants/motor_design.json (merged_design)。

    UI/CLI 传入 params (键名与 motor_design.json 一致, m/半径口径)。
    支持偏心 (eccentricity_mm, 转子圆系整体平移) 与 Halbach 分段充磁
    (magnetization='halbach', 每极 halbach_segments 段独立 Region: magnet_hb_k)。
    """
    DEFAULTS = dict(
        R_shaft=0.010, R_ri=0.026, mag_t=0.003, R_si=0.030,
        slot_depth=0.012, R_so=0.050,
        n_slots=12, n_poles=4,
        mag_half_deg=33.75, slot_half_deg=2.0, dtheta_max=2.5,
        eccentricity_mm=0.0, magnetization='radial', halbach_segments=0,
        rotor_angle_deg=0.0)

    def __init__(self, params=None):
        p = dict(self.DEFAULTS)
        p.update(params or {})
        for k in self.DEFAULTS:
            setattr(self, k, p[k])
        self.R_rm = self.R_ri + self.mag_t            # 磁体外半径 = 转子外半径
        self.g = self.R_si - self.R_rm                # 气隙长度
        # 注记10: ecc ≥ g 时定转子几何自交(气隙负/零), 任何网格手段都救不回 —
        # 实测 24s6p-hb5-ecc1.25 四种网格变体(加密/放粗/算法/尺寸)全部 >300s 死循环,
        # ecc 降至 0.75 后 2.7s 出网格 → 这是几何非法非求解器病态。
        if self.eccentricity_mm / 1000.0 >= self.g:
            raise ValueError(
                f"eccentricity_mm={self.eccentricity_mm} ≥ 气隙 g={self.g*1000:.3f}mm:"
                f" 定转子自交, 非法设计 (注记10)")
        # 注记11: 槽口角 2×slot_half_deg ≥ 齿距角 360°/n_slots 时相邻槽交叠、定子齿
        # 宽归零(几何自交), gmsh 虽能容忍出网格, 但磁导分布畸变 → Bg1 崩(全库 76/76
        # 违法例 Bg1<0.45T, 正常同形 0.73T)+THD 爆(35~55%) — 齿退化伪解。
        # 判定式: 2×sh < 360°/n_slots (硬约束); 设计推荐 2×sh ≤ 0.5×齿距角。
        tooth_pitch_deg = 360.0 / self.n_slots
        if 2.0 * self.slot_half_deg >= tooth_pitch_deg:
            raise ValueError(
                f"slot_half_deg={self.slot_half_deg} → 槽口角 {2*self.slot_half_deg}° "
                f"≥ 齿距角 {tooth_pitch_deg:.2f}° (n_slots={self.n_slots}):"
                f" 定子齿宽归零, 非法设计 (注记11)")
        self.ecc = self.eccentricity_mm / 1000.0      # 偏心距 (m), 转子圆系 +x 平移
        self.hb_seg = int(self.halbach_segments) if str(self.magnetization) == 'halbach' else 0
        self.rot = np.mod(float(self.rotor_angle_deg), 360.0)  # 剪切带步进旋转角
        self.edges = None
        self.angles = None

    # ---------- 角度网格 ----------
    def build_angles(self):
        slot_c = np.arange(self.n_slots) * (360.0 / self.n_slots)
        mag_c = np.arange(self.n_poles) * (360.0 / self.n_poles)
        edges = []
        for c in slot_c:
            edges += [c - self.slot_half_deg, c + self.slot_half_deg]
        for c in mag_c:
            edges += [c - self.mag_half_deg, c + self.mag_half_deg]
        # 注记10附注(实测回滚): 曾尝试把 halbach 段边界角加入 edges 使网格线
        # 对齐段边界, 但 90s12p-hb5-ecc05 复测显示该改动反而触发 gmsh 1D 自相交
        # (curves 302001/303001)死循环, 且使原本可救回的 dtheta_max=2.3 参数也失效。
        # 段边界歧义的可行解 = 保持整极弧细分 + dtheta_max 微调避让 (E 变体, 2.2s 过)。
        edges = sorted(set(round(np.mod(e, 360.0), 6) for e in edges))
        if edges[0] >= 360.0 - 1e-9:
            edges[0] = 0.0
        # 细分加密，保证角向单元角距 ≤ dtheta_max
        angles = []
        for i, e0 in enumerate(edges):
            e1 = edges[(i + 1) % len(edges)]
            width = e1 - e0
            if width <= 0:
                width += 360.0
            n_sub = max(1, int(np.ceil(width / self.dtheta_max)))
            for k in range(n_sub):
                angles.append(round(np.mod(e0 + width * k / n_sub, 360.0), 6))
        self.edges = edges
        self.angles = sorted(set(angles))
        return self

    def ring_radii(self):
        return [self.R_shaft, self.R_ri, self.R_rm, self.R_si,
                self.R_si + self.slot_depth, self.R_so]

    # ---------- 材料判定 ----------
    def magnet_state(self, theta_deg):
        """扇区中心角落在磁体弧内时: radial→('N'|'S'); halbach→段序int; 否则 None"""
        d = np.mod(np.asarray(theta_deg), 360.0)
        for pp in range(self.n_poles):
            c = pp * 360.0 / self.n_poles
            dd = np.mod(d - c + 180.0, 360.0) - 180.0
            if abs(dd) <= self.mag_half_deg - 1e-9:
                if self.hb_seg > 0:
                    # 段序: 极内 k (0..seg-1), 段宽均分 mag 弧
                    w = 2.0 * self.mag_half_deg / self.hb_seg
                    k = int(np.clip((dd + self.mag_half_deg) // w, 0, self.hb_seg - 1))
                    return pp * self.hb_seg + k
                return 'N' if pp % 2 == 0 else 'S'
        return None

    def slot_state(self, theta_deg):
        """返回 (槽号0..11, None) 或 (None, 'iron')：扇区中心是否在槽内"""
        d = np.mod(theta_deg, 360.0)
        k = int(round(d / (360.0 / self.n_slots))) % self.n_slots
        c = k * 360.0 / self.n_slots
        dd = np.mod(d - c + 180.0, 360.0) - 180.0
        if abs(dd) <= self.slot_half_deg - 1e-9:
            return k, None
        return None, 'iron'

    # 双层整距 12槽4极：上层相带 [A+,C-,B+,A-,C+,B-]×2；净槽电流分布 ±2Nc·i_ph
    PHASE_BELT = ['A+', 'C-', 'B+', 'A-', 'C+', 'B-']   # 60°相带基础序列(6槽周期)

    @classmethod
    def slot_group_name(cls, k, n_slots):
        """通用相带: 基础序列按 n_slots 周期循环 (12/24/48槽通用, 须为6的倍数)。"""
        if n_slots % 6 != 0:
            raise ValueError(f"n_slots={n_slots} 非6的倍数, 分数槽绕组暂不支持")
        belt = cls.PHASE_BELT[k % 6]
        return 'Slot' + belt[0] + ('p' if belt[1] == '+' else 'n')

    # ---------- .geo 生成 ----------
    def write_geo(self, path):
        self.build_angles()
        th = self.angles
        n_th = len(th)
        rings = self.ring_radii()
        w = []
        A = w.append

        A('// === 由 generate_geo.py 自动生成，勿手改 ===')
        A('lc_air = 0.0005;   lc_iron = 0.0012;  lc_shaft = 0.002;')
        A('Mesh.CharacteristicLengthExtendFromBoundary = 1;')
        A('')

        # 全局原点 (定子圆系圆心); 偏心时转子环 (ir<=ROTOR_RING_MAX) 圆心 +x 平移 ecc
        rotor_ring_max = 2   # rings: 0=shaft外圆 1=R_ri 2=R_rm 均属转子; >=R_si 属定子
        A('Point(1) = {0, 0, 0, lc_shaft};')
        A('// per-ring centers (偏心: 转子环圆心平移)')
        for ir in range(len(rings)):
            cx = self.ecc if ir <= rotor_ring_max else 0.0
            A(f'Point({900000 + ir + 1}) = {{{cx}, 0, 0, lc_shaft}};')
        A('// points on rings (转子环 ir<=2 角坐标 +rot: 剪切带步进旋转, 气隙带保形剪切)')
        for ir, r in enumerate(rings):
            lc = 'lc_air' if r in (self.R_rm, self.R_si) else 'lc_iron'
            cx = self.ecc if ir <= rotor_ring_max else 0.0
            for i, t in enumerate(th):
                tp = np.mod(t + self.rot, 360.0) if ir <= rotor_ring_max else t
                A(f'Point({1000*(ir+1) + i + 1}) = '
                  f'{{{cx} + {r}*Cos[{tp}*Pi/180], {r}*Sin[{tp}*Pi/180], 0, {lc}}};')

        # 每个圆环上的弧线（沿角度正方向）— 标签空间: 200000+ir*1000+i
        A('')
        A('// arcs on rings')
        for ir in range(len(rings)):
            base = 1000*(ir+1)
            ctr = 900000 + ir + 1
            for i in range(n_th):
                j = (i + 1) % n_th
                A(f'Circle({300000 + ir*1000 + i + 1}) = '
                  f'{{{base + i + 1}, {ctr}, {base + j + 1}}};')

        # 圆心已在最前定义
        A('')
        A('// spokes shaft->ring1')
        base0 = 1000
        for i in range(n_th):
            A(f'Line({990000 + i + 1}) = {{900000 + 1, {base0 + i + 1}}};')

        # 环带间的径向线
        A('')
        A('// radial lines between rings')
        for ir in range(len(rings) - 1):
            b0, b1 = 1000*(ir+1), 1000*(ir+2)
            for i in range(n_th):
                A(f'Line({400000 + ir*1000 + i + 1}) = '
                  f'{{{b0 + i + 1}, {b1 + i + 1}}};')

        # 扇区面
        A('')
        A('// sector surfaces (band × sector)')

        def sector_surface(ir, i, tag):
            """环带 ir(第ir与第ir+1圆之间) 的第 i 扇区面"""
            j = (i + 1) % n_th
            arc_hi = 300000 + (ir+1)*1000 + i + 1     # 外半径环弧 点i→点j
            arc_lo = 300000 + ir*1000 + i + 1         # 内半径环弧 点i→点j
            rad_j = 400000 + ir*1000 + j + 1          # 边缘j处径向线(内环→外环)
            rad_i = 400000 + ir*1000 + i + 1          # 边缘i处径向线(内环→外环)
            A(f'Line Loop({tag}) = {{{arc_lo}, {rad_j}, -{arc_hi}, -{rad_i}}};')
            A(f'Plane Surface({tag}) = {{{tag}}};')

        def shaft_sector_surface(i, tag):
            j = (i + 1) % n_th
            arc = 300000 + i + 1
            sp_i, sp_j = 990000 + i + 1, 990000 + j + 1
            A(f'Line Loop({tag}) = {{{arc}, -{sp_j}, {sp_i}}};')
            A(f'Plane Surface({tag}) = {{{tag}}};')

        tag = 100
        groups = {}
        outer_arcs = []

        def assign(name, st):
            groups.setdefault(name, []).append(st)

        # 轴盘
        for i in range(n_th):
            tag += 1
            shaft_sector_surface(i, tag)
            assign('shaft', tag)
        # 各环带
        n_bands = len(rings) - 1
        for ir in range(n_bands):
            r_mid = 0.5 * (rings[ir] + rings[ir+1])
            for i in range(n_th):
                tag += 1
                sector_surface(ir, i, tag)
                t_mid = 0.5 * (th[i] + (th[(i+1) % n_th] if th[(i+1) % n_th] > th[i]
                                        else th[(i+1) % n_th] + 360.0))
                band_names = ['rotor_iron', 'magnet', 'airgap', 'slot_band', 'stator_iron']
                name = band_names[ir]
                if name == 'magnet':
                    # 磁体随转子旋转: 标签按旋转后中角判定 (网格弧随之转动)
                    pol = self.magnet_state(t_mid + self.rot)
                    if pol is None:
                        name = 'rotor_iron'
                    elif isinstance(pol, int):
                        name = f'magnet_hb_{pol}'   # Halbach 分段 Region
                    else:
                        name = f'magnet_{pol}'
                elif name == 'slot_band':
                    k, is_iron = self.slot_state(t_mid)
                    name = (self.slot_group_name(k, self.n_slots)
                            if is_iron is None else 'stator_iron')
                assign(name, tag)
                if ir == n_bands - 1:      # 最外圈弧 → Dirichlet
                    j = (i + 1) % n_th
                    outer_arcs.append(300000 + (ir+1)*1000 + j + 1)

        # 物理组
        A('')
        A('// physical groups')
        phys_map = {}
        pid = 0
        for name, tags in groups.items():
            pid += 1
            phys_map[name] = pid
            A(f'Physical Surface("{name}", {pid}) = {{{",".join(map(str, tags))}}};')
        pid += 1
        A(f'Physical Curve("Outer", {pid}) = {{{",".join(map(str, outer_arcs))}}};')

        # 同步生成 GetDP 区域映射文件 (.pro 中 Region 必须用物理组标签号定义)
        pro_path = os.path.join(os.path.dirname(path), 'motor_groups.pro')
        g = ['// === 由 generate_geo.py 自动生成: 物理组标签 → GetDP 区域 ===', 'Group {']
        for name, tags in groups.items():
            g.append(f'  {name} = Region[{phys_map[name]}];')
        # 磁体段集: radial=N/S 交替; halbach=每段 magnet_hb_k — .pro 统一引用 MagSects
        mag_names = [n for n in groups if n.startswith('magnet_')]
        if mag_names:
            pids = sorted(phys_map[n] for n in mag_names)   # 物理组编号 (非面片tag!)
            g.append(f'  MagSects = Region[{{{",".join(map(str, pids))}}}]; '
                     f'// {len(mag_names)} 磁体段')
        g.append(f'  Outer = Region[{pid}];   // Dirichlet 外边界(曲线)')
        g.append('}')
        with open(pro_path, 'w') as f:
            f.write('\n'.join(g))
        # 名字→pid 映射 (run 链按名字取物理组, 杜绝编号硬编码)
        with open(os.path.join(os.path.dirname(path), 'groups.json'), 'w') as f:
            json.dump({name: pid for name, pid in phys_map.items()}, f, indent=1)
        A(f'// 区域映射已写入: {os.path.basename(pro_path)}')
        A('')
        A('Mesh.Algorithm = 6;')          # Frontal-Delaunay
        A('Mesh.MeshSizeExtendFromBoundary = 1;')
        A('Mesh 2;')

        with open(path, 'w') as f:
            f.write('\n'.join(w))
        return groups, n_th

    def summary(self):
        n = len(self.angles)
        counts = {'角向节点': n}
        return counts


class StatorStructMesh:
    """M90 定子实体结构网格 → CalculiX .inp (C3D8 六面体, 自由+约束模态)。

    口径 (modal_spec.py 为唯一事实源):
    - 几何: 定子内圆 R_si → 外圆 R_so 全环实体; 槽 (slot_half_deg 扇形, 槽深
      slot_depth) 为空单元剔除 → 齿+轭一体硅钢叠片芯; 轴向叠厚 L_stack。
    - 网格: 结构化六面体 — 角向 = 槽缘角线细分 (dtheta_max 口径),
      径向 = 齿带 1 层 + 轭带 (radial_layers-1) 层, 轴向 = axial_layers 层。
    - 材料 ELAM: 叠片正交各向异性 (modal_spec.lamination_equivalent),
      材料轴 = 全局轴 (叠片轴 = z), 无需 *Orientation。
    - 两步: Step1 自由模态 (无约束, 多求 6 阶覆盖刚体), Step2 约束模态
      (外圆 OUTER 固支 — 典型装机口径)。
    """

    def __init__(self, params=None, struct_cfg=None):
        p = dict(R_si=0.030, slot_depth=0.012, R_so=0.050,
                 n_slots=12, slot_half_deg=2.0, L_stack=0.1)
        p.update(params or {})
        for k in p:
            setattr(self, k, p[k])
        from modal_spec import load_nvh_struct, lamination_equivalent
        self.cfg = struct_cfg or load_nvh_struct()
        self.e_ip, self.e_ax, self.g12 = lamination_equivalent(
            self.cfg['nvh_struct_E_iron_pa'], self.cfg['nvh_struct_k_fill'],
            self.cfg['nvh_struct_nu_iron'])
        self.rho_eff = (self.cfg['nvh_struct_rho_iron']
                        * self.cfg['nvh_struct_k_fill'])

    # ---------- 角向线: 槽缘角 + 细分 (口径 nvh_struct_dtheta_max_deg) ----------
    def build_angles(self):
        sh = self.slot_half_deg
        edges = []
        for k in range(self.n_slots):
            c = k * 360.0 / self.n_slots
            edges += [np.mod(c - sh, 360.0), np.mod(c + sh, 360.0)]
        edges = sorted(set(round(e, 6) for e in edges))
        dmax = self.cfg['nvh_struct_dtheta_max_deg']
        angles = []
        for i, e0 in enumerate(edges):
            e1 = edges[(i + 1) % len(edges)]
            width = e1 - e0 if e1 > e0 else e1 + 360.0 - e0
            n_sub = max(1, int(np.ceil(width / dmax)))
            for kk in range(n_sub):
                angles.append(round(np.mod(e0 + width * kk / n_sub, 360.0), 6))
        return np.array(sorted(set(angles)))

    def in_slot(self, theta_deg):
        """扇区中心角是否在槽内 (槽 = 空单元)。"""
        d = np.mod(theta_deg, 360.0)
        k = int(round(d / (360.0 / self.n_slots))) % self.n_slots
        c = k * 360.0 / self.n_slots
        return abs(np.mod(d - c + 180.0, 360.0) - 180.0) <= self.slot_half_deg - 1e-9

    # ---------- 节点/单元编号 (径向 ir: 0..nr, 角向 ia, 轴向 iz: 0..nl) ----------
    def nid(self, ir, ia, iz, na):
        return 1 + (iz * (self.nr + 1) + ir) * na + ia

    def eid(self, ir, ia, iz, na):
        return 1 + (iz * self.nr + ir) * na + ia

    def build(self):
        """生成节点表/单元表 → (nodes, elems, na, nr, nl)。"""
        ang = self.build_angles()
        na = len(ang)
        r_tooth_hi = self.R_si + self.slot_depth
        radii = ([self.R_si + (r_tooth_hi - self.R_si) * (i + 1)
                  for i in range(1)] +
                 [r_tooth_hi + (self.R_so - r_tooth_hi) * (i + 1)
                  / max(1, self.cfg['nvh_struct_radial_layers'] - 1)
                  for i in range(self.cfg['nvh_struct_radial_layers'] - 1)])
        self.nr = len(radii)               # 单元层数 (ir: 0..nr-1)
        nl = int(self.cfg['nvh_struct_axial_layers'])
        zs = [self.L_stack * iz / nl for iz in range(nl + 1)]
        rr = [self.R_si] + radii
        nodes = {}
        for iz, z in enumerate(zs):
            for ir, r in enumerate(rr):
                for ia, t in enumerate(ang):
                    th = np.deg2rad(t)
                    nodes[self.nid(ir, ia, iz, na)] = (r * np.cos(th),
                                                       r * np.sin(th), z)
        elems = []                          # (eid, 8节点, mid_r, mid_th)
        void = 0
        for iz in range(nl):
            for ir in range(self.nr):
                for ia in range(na):
                    ja = (ia + 1) % na
                    n = [self.nid(ir, ia, iz, na), self.nid(ir + 1, ia, iz, na),
                         self.nid(ir + 1, ja, iz, na), self.nid(ir, ja, iz, na),
                         self.nid(ir, ia, iz + 1, na), self.nid(ir + 1, ia, iz + 1, na),
                         self.nid(ir + 1, ja, iz + 1, na), self.nid(ir, ja, iz + 1, na)]
                    t_mid = 0.5 * (ang[ia] + (ang[ja] if ang[ja] > ang[ia]
                                              else ang[ja] + 360.0))
                    if ir == 0 and self.in_slot(float(t_mid)):
                        void += 1
                        continue                    # 槽 = 空单元 (绕组不参与模态)
                    r_mid = 0.5 * (rr[ir] + rr[ir + 1])
                    elems.append((self.eid(ir, ia, iz, na), n, r_mid, float(t_mid)))
        self.nodes, self.elems, self.ang = nodes, elems, ang
        self.rr, self.zs = rr, zs          # 节点环半径/轴向层坐标 (振型采样用)
        return nodes, elems, na, self.nr, nl

    def aniso21_lines(self):
        """正交工程常数 → CalculiX TYPE=ANISO 21 常数卡 (等价数学口径: 同一柔度
        阵求逆; 本构建 ORTHO 读入段错误 + *Orientation 路径破坏求解, 故用全局
        轴 ANISO — 叠片轴 = 全局 z, 无需取向卡)。8 项/行 (132 字符行限)。

        材料轴 = 全局轴: E1=E2=E_ip (面内 x,y), E3=E_ax (叠片轴 z);
        Voigt 序 (11,22,33,12,13,23) 工程剪应变; 泊松耦合取 -nu/E_pair 均值
        (保 SPD, 与各向同性 nu 一致的工程近似口径)。
        """
        nu = self.cfg['nvh_struct_nu_iron']
        es = [self.e_ip, self.e_ip, self.e_ax]
        g = self.g12
        S = np.zeros((6, 6))
        for i in range(3):
            S[i, i] = 1.0 / es[i]
            for j in range(i + 1, 3):
                v = -nu * 2.0 / (es[i] + es[j])
                S[i, j] = S[j, i] = v
        for i in range(3, 6):
            S[i, i] = 1.0 / g
        D = np.linalg.inv(S)
        order = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2), (0, 3), (1, 3),
                 (2, 3), (3, 3), (0, 4), (1, 4), (2, 4), (3, 4), (4, 4),
                 (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
        vals = [f"{D[i, j]:.8g}" for i, j in order]
        return [','.join(vals[k:k + 8]) + (',' if k + 8 < 21 else '')
                for k in range(0, 21, 8)]

    def write_inp(self, path):
        """写 CalculiX .inp (Step1 自由 / Step2 外圆固支) → counts dict。"""
        nodes, elems, na, nr, nl = self.build()
        n_modes = int(self.cfg['nvh_struct_n_modes'])
        w = []
        A = w.append
        A('*Heading')
        A('M90 stator laminated-core modal analysis (auto: generate_geo.py --modal)')
        A('*Node')
        for nid in sorted(nodes):
            x, y, z = nodes[nid]
            A(f'{nid}, {x:.9g}, {y:.9g}, {z:.9g}')
        def nset_lines(ids):
            """CalculiX: 每行 ≤16 项; 仅续行以逗号结尾 (末行不可悬挂逗号)。"""
            ids = [str(i) for i in ids]
            lines = [','.join(ids[k:k + 16]) for k in range(0, len(ids), 16)]
            return [ln + ',' for ln in lines[:-1]] + [lines[-1]]

        A('*Nset, nset=ALLN')
        w.extend(nset_lines(sorted(nodes)))
        A('*Element, type=C3D8, Elset=ESTATOR')
        for eid, n, _, _ in elems:
            A(f'{eid}, ' + ', '.join(str(i) for i in n))
        r_so_nodes = [i for i, (x, y, z) in sorted(nodes.items())
                      if abs((x * x + y * y) ** 0.5 - self.R_so) < 1e-9]
        A('*Nset, nset=OUTER')
        w.extend(nset_lines(r_so_nodes))
        # 装机安装口径: 机座螺栓 4 点 (θ=45/135/225/315°) 弹性支承 (Spring1 接
        # 地, k=nvh_struct_k_mount_n_per_m, 螺栓联接典型 1e8..1e9 N/m)。整圈
        # 外圆固支/4 点刚性固支均为过约束 (基频被抬高一个量级, 物理不真)。
        na = len(self.ang)
        nl = int(self.cfg['nvh_struct_axial_layers'])
        k_mount = float(self.cfg['nvh_struct_k_mount_n_per_m'])
        mount = []
        for am in (45.0, 135.0, 225.0, 315.0):
            ia = int(np.argmin([min(abs(a - am), 360 - abs(a - am))
                                for a in self.ang]))
            for iz in range(nl + 1):
                mount.append(self.nid(self.nr - 1, ia, iz, na))
        mount = sorted(set(mount))
        A('*Nset, nset=MOUNT')
        w.extend(nset_lines(mount))
        spring_elsets = []
        for idof, tag in ((1, 'X'), (2, 'Y'), (3, 'Z')):
            tag = f'ESP{tag}'
            spring_elsets.append(tag)
            A(f'*Element, type=Spring1, Elset={tag}')
            w.extend(f'{90000 + idof * 10000 + k}, {n}'
                     for k, n in enumerate(mount))
            A(f'*Spring, Elset={tag}')
            # SPRING1: 第一行 dof (整数), 第二行刚度 (实数, 必带小数点)
            A(f'{idof}')
            A(f'{k_mount:.4E}')
        A('*Elset, Elset=ESPRING')
        w.extend(nset_lines(spring_elsets))
        spring_eids = [90000 + d * 10000 + k
                       for d in (1, 2, 3) for k in range(len(mount))]
        A('*Solid Section, Elset=ESTATOR, Material=ELAM')
        A('*Material, Name=ELAM')
        A('*Elastic, Type=Aniso')
        w.extend(self.aniso21_lines())
        A('*Density')
        A(f'{self.rho_eff:.6g}')
        A('** ---- Step1: 自由模态 (多求6阶覆盖刚体; 移除安装弹簧) ----')
        A('*Step')
        A('*Model Change, Type=Element, Remove')
        A('ESPRING')
        A('*Frequency')
        A(f'{n_modes + 6}')
        A('*Node Print, Nset=ALLN')
        A('U')
        A('*End Step')
        A('** ---- Step2: 约束模态 (机座 4 点弹性支承, 装机口径) ----')
        A('*Step')
        A('*Model Change, Type=Element, Add=With Strain')
        A('ESPRING')
        A('*Frequency')
        A(f'{n_modes}')
        A('*Node Print, Nset=ALLN')
        A('U')
        A('*End Step')
        with open(path, 'w') as f:
            f.write('\n'.join(w) + '\n')
        return {'inp': path, 'n_nodes': len(nodes), 'n_elem': len(elems),
                'n_void_slot_elem': na * nr * nl - len(elems),
                'angles': na, 'radial_layers': nr, 'axial_layers': nl,
                'e_ip_pa': self.e_ip, 'e_ax_pa': self.e_ax,
                'rho_eff_kgm3': self.rho_eff,
                'n_modes_free': n_modes + 6, 'n_modes_constrained': n_modes}


class MechStructMesh:
    """M93 转子/磁钢实体结构网格 (CalculiX C3D8, 静力强度)。

    复用 M90 StatorStructMesh 思路: 角向 = 磁体边缘角 + 细分 (mech_dtheta_max_deg
    口径), 径向 = 铁芯环/磁钢环各 radial_layers 层, 轴向 = axial_layers 层。

    口径 (mech_spec.py 为唯一事实源):
    - 几何: 转子铁芯环 R_shaft..R_ri + 磁钢环 R_ri..R_rm 双带实体; 轴孔内表面
      自由 (轴联接应力不构成本链目标), 全 360° 环模型。
    - 单元: C3D8 六面体, 节点序 = 面片1(内圆柱面)-面片2(外圆柱面) 排布,
      使过盈界面压/内压可按 *Dload P1/P2 直接施加。
    - 材料: 各向同性 (铁芯/磁钢, mech_E/nux/rho), 实心口径 (强度非叠片等效,
      与 M90 模态叠片 ANISO 口径区分, 已在 mech_spec 披露)。
    """

    def __init__(self, params=None, mech_cfg=None, bands=None):
        p = dict(R_shaft=0.010, R_ri=0.026, mag_t=0.003,
                 n_poles=4, mag_half_deg=33.75, L_stack=0.1)
        p.update(params or {})
        for k in p:
            setattr(self, k, p[k])
        self.R_rm = self.R_ri + self.mag_t
        from mech_spec import load_mech_struct
        self.cfg = mech_cfg or load_mech_struct()
        # 带: (组名, r0, r1); 缺省 = 铁芯环 + 磁钢环
        self.bands = bands or [('iron', self.R_shaft, self.R_ri),
                               ('magnet', self.R_ri, self.R_rm)]
        self.L = self.L_stack

    def build_angles(self):
        """磁体边缘角 + 细分, 口径 mech_dtheta_max_deg。"""
        dmax = self.cfg['mech_dtheta_max_deg']
        edges = []
        for pp in range(int(self.n_poles)):
            c = pp * 360.0 / self.n_poles
            edges += [np.mod(c - self.mag_half_deg, 360.0),
                      np.mod(c + self.mag_half_deg, 360.0)]
        edges = sorted(set(round(e, 6) for e in edges))
        angles = []
        for i, e0 in enumerate(edges):
            e1 = edges[(i + 1) % len(edges)]
            width = e1 - e0 if e1 > e0 else e1 + 360.0 - e0
            n_sub = max(1, int(np.ceil(width / dmax)))
            for kk in range(n_sub):
                angles.append(round(np.mod(e0 + width * kk / n_sub, 360.0), 6))
        return np.array(sorted(set(angles)))

    def build_radii(self):
        """各带径向均分 n 层 → 全部节点圆半径 (含内外表面)。"""
        n = int(self.cfg['mech_radial_layers_per_band'])
        rr = [self.bands[0][1]]
        for _, r0, r1 in self.bands:
            rr += [r0 + (r1 - r0) * (k + 1) / n for k in range(n)]
        return rr

    # 节点/单元编号: 与 StatorStructMesh 同式 (ir: 0..nr, ia: 0..na-1, iz: 0..nl)
    def nid(self, ir, ia, iz, na):
        return 1 + (iz * (self.nr + 1) + ir) * na + ia

    def eid(self, ir, ia, iz, na):
        return 1 + (iz * self.nr + ir) * na + ia

    def build(self):
        """生成节点/单元表 → (nodes, elems, ang, rr, zs)。

        elems: (eid, [8节点], band_name, ir_band, r_mid, th_mid_deg);
        单元节点序: n1..n4 = 内圆柱面 (θ0z0→θ1z0→θ1z1→θ0z1), n5..n8 = 外圆柱面
        同序 → face1=内面, face2=外面 (P1/P2 面压可直接施加)。
        """
        ang = self.build_angles()
        na = len(ang)
        rr = self.build_radii()
        self.nr = len(rr) - 1
        nl = int(self.cfg['mech_axial_layers'])
        zs = [self.L * iz / nl for iz in range(nl + 1)]
        nodes = {}
        for iz, z in enumerate(zs):
            for ir, r in enumerate(rr):
                for ia, t in enumerate(ang):
                    th = np.deg2rad(t)
                    nodes[self.nid(ir, ia, iz, na)] = (r * np.cos(th),
                                                       r * np.sin(th), z)
        elems = []
        band_of_ir = {}
        for ib, (bname, _, _) in enumerate(self.bands):
            for k in range(int(self.cfg['mech_radial_layers_per_band'])):
                band_of_ir[ib * int(self.cfg['mech_radial_layers_per_band'])
                           + k] = (bname, k)
        for iz in range(nl):
            for ir in range(self.nr):
                for ia in range(na):
                    ja = (ia + 1) % na
                    n = [self.nid(ir, ia, iz, na), self.nid(ir, ja, iz, na),
                         self.nid(ir, ja, iz + 1, na), self.nid(ir, ia, iz + 1, na),
                         self.nid(ir + 1, ia, iz, na), self.nid(ir + 1, ja, iz, na),
                         self.nid(ir + 1, ja, iz + 1, na),
                         self.nid(ir + 1, ia, iz + 1, na)]
                    bname, k = band_of_ir[ir]
                    t0, t1 = ang[ia], ang[ja]
                    t_mid = 0.5 * (t0 + (t1 if t1 > t0 else t1 + 360.0))
                    r_mid = 0.5 * (rr[ir] + rr[ir + 1])
                    elems.append((self.eid(ir, ia, iz, na), n, bname, k,
                                  float(r_mid), float(np.mod(t_mid, 360.0))))
        self.nodes, self.elems, self.ang = nodes, elems, ang
        self.rr, self.zs = rr, zs
        return nodes, elems, ang, rr, zs


if __name__ == '__main__':
    import sys, json
    # 输出目录跟随 --params 所在目录 (workdir 隔离: 并行批跑每模型独立)
    out_dir = os.path.dirname(os.path.abspath(__file__))
    params = None
    if '--params' in sys.argv:
        _pf = sys.argv[sys.argv.index('--params') + 1]
        params = json.load(open(_pf, encoding='utf-8'))
        params = {k: v for k, v in params.items() if not k.startswith('_')}
        out_dir = os.path.dirname(os.path.abspath(_pf))
    if '--modal' in sys.argv:
        # M90 定子实体结构网格 → CalculiX .inp (自由+约束模态两步)
        sm = StatorStructMesh(params)
        info = sm.write_inp(os.path.join(out_dir, 'motor_stator_modal.inp'))
        print('.inp 已生成:', info)
    elif '--mech' in sys.argv:
        # M93 转子/磁钢实体结构网格 (静力强度链由 run_mech_ccx.py 装配)
        mm = MechStructMesh(params)
        nodes, elems, ang, rr, zs = mm.build()
        print(f'MechStructMesh: {len(nodes)} 节点, {len(elems)} C3D8, '
              f'{len(ang)} 角向, 半径层 {rr}')
    else:
        gw = SectorGeoWriter(params)
        geo_path = os.path.join(out_dir, 'motor_sector.geo')
        groups, n_th = gw.write_geo(geo_path)
        print(f'.geo 已生成: {geo_path}')
        print(f'角向节点数: {n_th}, 气隙: {gw.g*1e3:.2f} mm')
        for name, tags in sorted(groups.items()):
            print(f'  {name}: {len(tags)} 面片')
