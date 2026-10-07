// === 由 generate_geo.py 自动生成: 物理组标签 → GetDP 区域 ===
Group {
  shaft = Region[1];
  rotor_iron = Region[2];
  magnet_N = Region[3];
  magnet_S = Region[4];
  airgap = Region[5];
  SlotAp = Region[6];
  stator_iron = Region[7];
  SlotCn = Region[8];
  SlotBp = Region[9];
  SlotAn = Region[10];
  SlotCp = Region[11];
  SlotBn = Region[12];
  MagSects = Region[{3,4}]; // 2 磁体段
  Outer = Region[13];   // Dirichlet 外边界(曲线)
}