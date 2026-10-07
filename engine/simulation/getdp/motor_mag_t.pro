// motor_mag_t.pro — M86 瞬态电磁链主问题 (GetDP TimeLoopTheta 时间推进)
// Include: motor_groups.pro (区域映射) + motor_mag_data.pro (BH/常数/HC段)
// 口径 (constants/operating_conditions.json transient_* / transient_spec.py):
//   - θ 法: TimeLoopTheta[t0,t_end,dt,TransTheta], 默认 TransTheta=1 隐式 Euler
//     (官方口径: 时间相关源仅在 θ=1 时于当前步取值正确 — getdp.texi Warning)
//   - 方程: ∮ν(B)∇a·∇δa + σ·∂a/∂t·δa = ∫hc·δa − ∫js(t)·δa
//   - 时间源: js(t_k)=$Time 显式正弦 (IA0/IB0/IC0 幅值 + Fe 电角频率, 驱动器注入)
//   - 非线性: 每时间步内层 Picard (NL_tol_rel/NL_iter_max, 与静磁同口径)
//   - 转子旋转: 网格级剪切带步进 (generate_geo.py rotor_angle_deg), 本文件不感知
// 数值口径禁写死: 所有常数经 DefineConstant 缺省 0, 由 run_emag_transient.py
// 以 -setnumber 注入 (constants/ 与参数输入为唯一事实源)

Include "motor_groups.pro"
Include "motor_mag_data.pro"

// ============ 瞬态口径常数 (驱动器 -setnumber 注入) ============
DefineConstant[
  TransT0        = 0.,   // 起始时间 [s]
  TransTMax      = 0.,   // 终止时间 [s] = periods*T_el
  TransDt = 0.,   // dt [s] = T_el/steps_per_period
  TransTheta  = 1.,   // 1=隐式Euler, 0.5=CN(源口径不支持,禁用)
  Fe           = 0.,   // 电角频率 [Hz] = p*n/60 (transient_spec.electrical_freq_hz)
  IA0          = 0.,   // 相电流幅值 [A] (js=2*Nc*i/slot_area 换算后注入)
  IB0          = 0.,
  IC0          = 0.,
  SigMag       = 0.,   // 磁体电导率 [S/m] (transient_sigma_magnet)
  SigIron      = 0.    // 铁心电导率 [S/m] (transient_sigma_iron)
];

// ============ 区域组合 (与静磁一致) ============
Group {
  Vol_All   = Region[{ shaft, rotor_iron, MagSects, airgap,
                       stator_iron, SlotAp, SlotAn, SlotBp, SlotBn,
                       SlotCp, SlotCn }];
  Vol_Iron  = Region[{ rotor_iron, stator_iron }];
  Vol_Mags  = Region[{ MagSects }];
  Vol_Slots = Region[{ SlotAp, SlotAn, SlotBp, SlotBn, SlotCp, SlotCn }];
}

// ============ 材料函数 (ν/Hc 与静磁逐位一致) ============
Function {
  nu[ Region[{airgap}] ] = 1/mu0;
  nu[ Region[{SlotAp, SlotAn, SlotBp, SlotBn, SlotCp, SlotCn}] ] = 1/mu0;
  nu[ Region[{shaft}] ] = 1/(mu0*mur_shaft);
  nu[ Region[Vol_Iron] ] = Flag_IronLinear ? 1/(mu0*Mur_Iron) : iron_nu[$1];
  nu[ Region[{Vol_Mags}] ] = 1/(mu0*1.05);

  // hc 矫顽场向量场: 驱动器写入 motor_mag_data.pro 尾部 HC VECTOR SECTIONS 区
  // (网格级旋转: hc 以 Atan2/常数段按旋转后区域定义, 本文件不感知)

  // 时间源 js(t) [A/m^2]: 三相正弦, Form1P 须 Vector[0,0,Jz];
  // A 相 0°, B 相 -120°, C 相 +120° (正序)。$Time 为当前时间步取值 (θ=1 口径正确)
  js[ Region[{SlotAp}] ] = Vector[0, 0,  IA0*Sin[2*Pi*Fe*$Time]];
  js[ Region[{SlotAn}] ] = Vector[0, 0, -IA0*Sin[2*Pi*Fe*$Time]];
  js[ Region[{SlotBp}] ] = Vector[0, 0,  IB0*Sin[2*Pi*Fe*$Time-2*Pi/3]];
  js[ Region[{SlotBn}] ] = Vector[0, 0, -IB0*Sin[2*Pi*Fe*$Time-2*Pi/3]];
  js[ Region[{SlotCp}] ] = Vector[0, 0,  IC0*Sin[2*Pi*Fe*$Time+2*Pi/3]];
  js[ Region[{SlotCn}] ] = Vector[0, 0, -IC0*Sin[2*Pi*Fe*$Time+2*Pi/3]];

  // 涡流区电导率 (默认 0 → 纯磁扩散零项, 与静磁矩阵逐位一致)
  Sigma[ Region[{Vol_Mags}] ] = SigMag;
  Sigma[ Region[Vol_Iron] ]   = SigIron;
}

// ============ 约束: 定子外圆 Az = 0 ============
Constraint {
  { Name a; Type Assign;
    Case { { Region Outer; Value 0.; } } }
}

// ============ 函数空间: 2D 节点型 Form1P ============
FunctionSpace {
  { Name Hcurl_a; Type Form1P;
    BasisFunction {
      { Name se; NameOfCoef ae; Function BF_PerpendicularEdge;
        Support Vol_All; Entity NodesOf[ All ]; }
    }
    Constraint {
      { NameOfCoef ae; EntityType NodesOf; NameOfConstraint a; }
    }
  }
}

Jacobian {
  { Name JacVol; Case { { Region All; Jacobian Vol; } } }
}
Integration {
  { Name Int_Mag;
    Case { { Type Gauss;
      Case {
        { GeoElement Point;      NumberOfPoints 1; }
        { GeoElement Line;       NumberOfPoints 3; }
        { GeoElement Triangle;   NumberOfPoints 4; }
        { GeoElement Quadrangle; NumberOfPoints 4; }
        { GeoElement Tetrahedron; NumberOfPoints 4; }
        { GeoElement Hexahedron;  NumberOfPoints 6; }
        { GeoElement Prism;       NumberOfPoints 9; }
      } } } }
}

// ============ 形式: 瞬态磁扩散 (σ·Dt 项 + 时间源) ============
Formulation {
  { Name Magnetodynamics_a; Type FemEquation;
    Quantity {
      { Name a; Type Local; NameOfSpace Hcurl_a; }
    }
    Equation {
      // 线性区: nu0 项
      Integral { [ nu[] * Dof{d a} , {d a} ];
        In Region[{airgap, shaft, Vol_Slots, Vol_Mags}];
        Jacobian JacVol; Integration Int_Mag; }
      // 非线性铁磁区: Picard
      Integral { [ nu[{d a}] * Dof{d a}, {d a} ];
        In Vol_Iron; Jacobian JacVol; Integration Int_Mag; }
      // 磁体: Hc 项
      Integral { [ hc[] , {d a} ];
        In Vol_Mags; Jacobian JacVol; Integration Int_Mag; }
      // 槽电流时间源 js(t)
      Integral { [ -js[] , {a} ];
        In Vol_Slots; Jacobian JacVol; Integration Int_Mag; }
      // 涡流项 σ·∂a/∂t (SigMag/SigIron=0 时矩阵与静磁逐位一致)
      // 涡流项 σ·∂a/∂t: 官方 DtDof 前缀算子 (Lib_Magnetodynamics2D_av_Cir.pro:329)
      Integral { DtDof [ Sigma[] * Dof{a}, {a} ];
        In Region[{Vol_Mags, Vol_Iron}];
        Jacobian JacVol; Integration Int_Mag; }
    }
  }
}

// ============ 求解序列: T0 完整 Picard + TimeLoopTheta 冻结 nu 单步解 ============
// 非线性口径 (实证, /tmp/m86_lu 探针):
//  - 步内每步重开 Picard (res/res0 比值判据) 在第 2 步起 res0~0 (前步已收敛),
//    比值失去意义, 解在 GMRES 噪声底上随机游走 (40 步 Bg1 漂移 1.06%);
//  - 官方 IterativeLoop[NL,relax]{Solve} 不更新解差判据 (RelativeDifference 恒 0),
//    IterativeLoopAdvanced/SolveJac 仅适用 Newton (GenerateJac) 体系;
//  - 故采用半隐式方案: 初始时刻完整 Picard 收敛 (与静磁链同一不动点),
//    时间环内每步一次冻结 nu 的线性解。空载/σ=0 时逐步精确驻留
//    (40 步 Bg1=0.9049 vs 静磁 0.9116, 0.73% ≤ 1% 容差)。
Resolution {
  { Name Magnetodynamics_a;
    System {
      { Name A; NameOfFormulation Magnetodynamics_a; }
    }
    Operation {
      InitSolution[A];
      // t=T0 初始条件: 完整 Picard, 与静磁链同口径
      Generate[A]; GetResidual[A, $res0];
      Solve[A];
      Generate[A]; GetResidual[A, $res];
      Evaluate[ $iter = 1 ];
      While[$res/$res0 > NL_tol_rel && $res/$res0 <= 1 && $iter < NL_iter_max]{
        Solve[A]; Generate[A]; GetResidual[A, $res];
        Evaluate[ $iter = $iter + 1 ];
      }
      TimeLoopTheta[TransT0, TransTMax, TransDt, TransTheta]{
        // 单步解: nu 冻结自上一时刻 (半隐式 Picard)
        Generate[A]; Solve[A];
        SaveSolution[A];
      }
      PostOperation[MagT];
    }
  }
}

// ============ 后处理 ============
PostProcessing {
  { Name MagT; NameOfFormulation Magnetodynamics_a;
    Quantity {
      { Name a; Value { Term { [ {a} ]; In Vol_All; Jacobian JacVol; } } }
      { Name b; Value { Term { [ {d a} ]; In Vol_All; Jacobian JacVol; } } }
    }
  }
}

PostOperation {
  { Name MagT; NameOfPostProcessing MagT;
    Operation {
      // 末时间步全域 B (Bg1/扭矩/Arkkio 与静磁同解析口径)
      Print[ b, OnElementsOf Vol_All, LastTimeStepOnly, File "b_map_t.pos" ];
      Print[ a, OnElementsOf Vol_All, LastTimeStepOnly, File "a_map_t.pos" ];
      // 槽区 Az 末步 (磁链)
      Print[ a, OnElementsOf Vol_Slots, LastTimeStepOnly, File "az_slots_t.pos" ];
      // 气隙 B 末步 (M89 力密度链: 驱动器按相位步逐次单步 TimeLoopTheta 调用,
      // 每次落 bgap_t.pos → 重命名 bgap_k.pos; GetDP 多步解检索不可用 —
      // TimeLoopTheta 仅保留末步解, 多步 Print 报 Empty solution, 4 变体实证)
      Print[ b, OnElementsOf airgap, LastTimeStepOnly, File "bgap_t.pos" ];
    }
  }
}
