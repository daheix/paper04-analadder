// A2 阶梯验证网格: 均匀横向磁化实心圆柱 + 外域空气 (2D)
// 物理标签: 3=pm 磁柱 (M = M*xhat 常矢量), 2=air, 外边界 A=0
// 解析参考 (R=Dirichlet 半径): A2=(M/2)(1-b^2/R^2) → B_in = mu0*M*(1+b^2/R^2)/2 沿 x
R = 0.15;  // 外边界半径 (Dirichlet A=0)
b = 0.05;  // 磁柱半径 (R = 3b → 修正因子 (1+1/9)/2)
lc = 0.002;
Point(1) = {0, 0, 0, lc};
Point(2) = {b, 0, 0, lc}; Point(3) = {-b, 0, 0, lc};
Point(4) = {0, b, 0, lc}; Point(5) = {0, -b, 0, lc};
Circle(1) = {2, 1, 4}; Circle(2) = {4, 1, 3}; Circle(3) = {3, 1, 5}; Circle(4) = {5, 1, 2};
Point(6) = {R, 0, 0, 1.5*lc}; Point(7) = {-R, 0, 0, 1.5*lc};
Point(8) = {0, R, 0, 1.5*lc}; Point(9) = {0, -R, 0, 1.5*lc};
Circle(5) = {6, 1, 8}; Circle(6) = {8, 1, 7}; Circle(7) = {7, 1, 9}; Circle(8) = {9, 1, 6};
Curve Loop(1) = {5, 6, 7, 8};
Curve Loop(2) = {1, 2, 3, 4};
Plane Surface(2) = {1, 2}; // air
Plane Surface(3) = {2};    // pm cylinder
Physical Surface("air") = {2};
Physical Surface("pm") = {3};
Physical Curve("outer") = {5, 6, 7, 8};
Mesh.Algorithm = 6;
Mesh 2;
Save "test_cyl.msh";
