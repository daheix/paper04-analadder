// A1 阶梯验证网格: 中心导体圆盘 + 外域空气 (2D, z 向单位厚度)
// 物理标签: 1=conductor(r<rc, J=I/(pi*rc^2)), 2=air, 外边界 A=0
R = 0.10;   // 外边界半径 (Dirichlet A=0)
rc = 0.005; // 导体半径
lc = 0.0022;
Point(1) = {0, 0, 0, lc};
Point(2) = {rc, 0, 0, lc/3};
Point(3) = {-rc, 0, 0, lc/3};
Point(4) = {0, rc, 0, lc/3};
Point(5) = {0, -rc, 0, lc/3};
Circle(1) = {2, 1, 4};
Circle(2) = {4, 1, 3};
Circle(3) = {3, 1, 5};
Circle(4) = {5, 1, 2};
Point(6) = {R, 0, 0, 1.4*lc};
Point(7) = {-R, 0, 0, 1.4*lc};
Point(8) = {0, R, 0, 1.4*lc};
Point(9) = {0, -R, 0, 1.4*lc};
Circle(5) = {6, 1, 8};
Circle(6) = {8, 1, 7};
Circle(7) = {7, 1, 9};
Circle(8) = {9, 1, 6};
Curve Loop(1) = {5, 6, 7, 8};
Curve Loop(2) = {1, 2, 3, 4};
Plane Surface(2) = {1, 2}; // air (带洞)
Plane Surface(1) = {2};    // conductor
Physical Surface("conductor") = {1};
Physical Surface("air") = {2};
Physical Curve("outer") = {5, 6, 7, 8};
Mesh.Algorithm = 6;
Mesh 2;
