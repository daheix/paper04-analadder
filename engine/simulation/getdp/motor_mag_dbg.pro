// motor_mag.pro — 4极12槽 SPM 电机 2D 磁静力学主问题
// Include: motor_mag_data.pro (区域/BH数据/常数定义)
// 求解: Picard 非线性迭代 (官方模板 Lib_Magnetostatics_a_phi.pro 的 Else 分支)
// 后处理: 气隙圆周 B 采样(Maxwell应力扭矩) + 槽区 Az(磁链) + 全域 B(Bmax/Arkkio)

Include "motor_groups.pro"
Include "motor_mag_data.pro"

// ============ 区域组合 ============
Group {
  Vol_All   = Region[{ shaft, rotor_iron, magnet_N, magnet_S, airgap,
                       stator_iron, SlotAp, SlotAn, SlotBp, SlotBn,
                       SlotCp, SlotCn }];
  Vol_Iron  = Region[{ rotor_iron, stator_iron }];
  Vol_Mags  = Region[{ magnet_N, magnet_S }];
  Vol_Slots = Region[{ SlotAp, SlotAn, SlotBp, SlotBn, SlotCp, SlotCn }];
}

// ============ 区域上的材料函数 ============
Function {
  // 线性区: 气隙/槽(铜+槽楔, mu_r≈1) / 轴(mu_r=700)
  nu[ Region[{airgap}] ] = 1/mu0;
  nu[ Region[{SlotAp, SlotAn, SlotBp, SlotBn, SlotCp, SlotCn}] ] = 1/mu0;
  nu[ Region[{shaft}] ] = 1/(mu0*mur_shaft);
  // 非线性铁磁区 (Picard: nu 取当前解的 B)
  nu[ Region[Vol_Iron] ] = Flag_IronLinear ? 1/(mu0*Mur_Iron) : iron_nu[$1];
  // 磁体: mu_r = 1.05
  nu[ Region[{magnet_N, magnet_S}] ] = 1/(mu0*1.05);

  // 磁体矫顽场矢量: 径向充磁, N 极(0°,180°)向外, S 极(90°,270°)反向
  // 注: GetDP 将非 Dof 的 Integral 项移到方程右侧并变号,
  //     故此处 N 极取 -Hc 使解出的 B 在 N 极向外 (H = nu*B - Hc 约定).
  hc[ Region[{magnet_N}] ] = Vector[-Hc_mag*Cos[Atan2[Y[],X[]]],
                                    -Hc_mag*Sin[Atan2[Y[],X[]]], 0. ];
  hc[ Region[{magnet_S}] ] = Vector[ Hc_mag*Cos[Atan2[Y[],X[]]],
                                     Hc_mag*Sin[Atan2[Y[],X[]]], 0. ];

  // 槽电流密度 Jz [A/m^2]: 各相槽净导体 2*Nc, 由驱动器直接给 Jz
  js[ Region[{SlotAp}] ] = IA;   js[ Region[{SlotAn}] ] = IAn;
  js[ Region[{SlotBp}] ] = IB;   js[ Region[{SlotBn}] ] = IBn;
  js[ Region[{SlotCp}] ] = IC;   js[ Region[{SlotCn}] ] = ICn;
}

// ============ 约束: 定子外圆 Az = 0 ============
Constraint {
  { Name a; Type Assign;
    Case { { Region Outer; Value 0.; } } }
}

// ============ 函数空间: 2D 节点型 Form1P (Az 未知量) ============
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

// ============ 形式 ============
Formulation {
  { Name Magnetostatics_a; Type FemEquation;
    Quantity {
      { Name a; Type Local; NameOfSpace Hcurl_a; }
    }
    Equation {
      // 线性区: nu0 项
      Integral { [ nu[] * Dof{d a} , {d a} ];
        In Region[{airgap, shaft, Vol_Slots, magnet_N, magnet_S}];
        Jacobian JacVol; Integration Int_Mag; }
      // 非线性铁磁区: Picard
      Integral { [ nu[{d a}] * Dof{d a}, {d a} ];
        In Vol_Iron; Jacobian JacVol; Integration Int_Mag; }
      // 磁体: Hc 项
      Integral { [ hc[] , {d a} ];
        In Vol_Mags; Jacobian JacVol; Integration Int_Mag; }
      // 槽电流源
      Integral { [ -js[] , {a} ];
        In Vol_Slots; Jacobian JacVol; Integration Int_Mag; }
    }
  }
}

// ============ 求解序列 ============
Resolution {
  { Name Magnetostatics_a;
    System {
      { Name A; NameOfFormulation Magnetostatics_a; }
    }
    Operation {
      InitSolution[A];
      Generate[A]; Solve[A];
      Generate[A]; GetResidual[A, $res0];
      Evaluate[ $res = $res0, $iter = 0 ];
      While[$res/$res0 > NL_tol_rel && $res/$res0 <= 1 && $iter < NL_iter_max]{
        Solve[A]; Generate[A]; GetResidual[A, $res];
        Evaluate[ $iter = $iter + 1 ];
      }
      SaveSolution[A];
      PostOperation[Mag];
    }
  }
}

// ============ 后处理 ============
PostProcessing {
  { Name Mag; NameOfFormulation Magnetostatics_a;
    Quantity {
      { Name a; Value { Term { [ {a} ]; In Vol_All; Jacobian JacVol; } } }
      { Name b; Value { Term { [ {d a} ]; In Vol_All; Jacobian JacVol; } } }
      { Name js; Value { Term { [ js[] ]; In Vol_Slots; Jacobian JacVol; } } }
    }
  }
}

PostOperation {
  { Name Mag; NameOfPostProcessing Mag;
    Operation {
      // 全域 B 场图(含单元中心值 → Bmax 与 Arkkio 积分用)
      Print[ b, OnElementsOf Vol_All, File "b_map.pos" ];
      Print[ a, OnElementsOf Vol_All, File "a_map.pos" ];
      // 槽区 Az (磁链积分用)
      Print[ js, OnElementsOf Vol_Slots, File "js_check.pos" ];
      // 气隙圆周 B 由驱动器从 b_map.pos 插值(Maxwell 应力扭矩)
    }
  }
}
