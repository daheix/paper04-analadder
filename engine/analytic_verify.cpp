// analytic_verify.cpp — 电机电磁链路解析验证阶梯 (纯 C++ 求解层, Eigen LLT 直接法)
//
// 模式:
//   coil <msh> <I_A>            A1: 中心导体圆盘, FEM B_theta vs mu0*I/(2*pi*r)
//   cyl  <msh> <M_A/m>          A2: 均匀横向磁化实心圆柱, FEM vs 有限边界解析解
//   motor <msh> <bmap.pos>      B : 无载气隙 Fourier 谱 + Carter 修正 1D 磁路
//
// 解析基准推导:
//   A1: 圆对称 A_z(r)=mu0*I/(2π)·ln(C/r) → B_theta = mu0*I/(2πr) 对任意 Dirichlet 外界精确成立。
//   A2: 均匀 M=M·x̂ 圆柱(半径 b) + 空气(R 边界 A=0), 标势谐波解 (H=−∇φ, B=μ0(−∇φ+M)):
//       磁柱内 φ=A2·r·cosθ, A2=(M/2)(1+b²/R²) → B_in = mu0*M*(1-b²/R²)/2 · x̂ (R→∞ 时 mu0*M/2)
//       空气域 φ=A4·(1/r - r/R²)·cosθ, A4=M·b²/2 (H 场量纲)
//         B_r = μ0·A4·(1/r²+1/R²)·cosθ,  B_θ = μ0·A4·(1/r²-1/R²)·sinθ
//   B : Carter γ=(b0/g)²/(5+b0/g), k_c=ts/(ts-γg), Bg=Br·hm/(hm+μr·kc·g)
//
// P1 FEM 内核与 vw_arbiter.cpp 同源: K_ij=ν(b_i b_j+c_i c_j)/(2|twoA|),
//   PM 源 f_i=0.5(hx·c_i-hy·b_i)·sgn(twoA), B 后处理 Bx=ΣA_i c_i/twoA, By=-ΣA_i b_i/twoA。
// 编译: g++ -std=c++23 -O2 -I<proj>/bin/third_party/eigen/include/eigen5 -include cstring \
//         -o analytic_verify analytic_verify.cpp
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

using namespace std;
static constexpr double MU0 = 4e-7 * M_PI;

struct Tri { int n[3]; int phys; };

struct Msh {
    vector<double> x, y;      // 节点坐标 (0-based)
    vector<Tri> tri;          // 三角形 (节点索引 + 物理标签)
    map<string, int> name2tag;  // $PhysicalNames: 名称 → 物理标签
    // ---- msh2 读取: $Nodes "id x y z" / $Elements "id etype ntags tags... n1 n2 n3"
    static Msh read(const string& path) {
        FILE* f = fopen(path.c_str(), "r");
        if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); exit(1); }
        Msh m; char line[4096];
        enum Sec { NONE, NAMES, NODES, ELEMS } sec = NONE;
        while (fgets(line, sizeof line, f)) {
            if (!strncmp(line, "$PhysicalNames", 14)) { sec = NAMES; continue; }
            if (!strncmp(line, "$EndPhysicalNames", 17)) { sec = NONE; continue; }
            if (!strncmp(line, "$Nodes", 6)) { sec = NODES; continue; }
            if (!strncmp(line, "$EndNodes", 9)) { sec = NONE; continue; }
            if (!strncmp(line, "$Elements", 9)) { sec = ELEMS; continue; }
            if (!strncmp(line, "$EndElements", 12)) { sec = NONE; continue; }
            if (sec == NONE) continue;
            if (sec == NAMES) {
                // gmsh 4.x 输出 msh2 时会按 (维度, 顺序) 重映射物理标签, 须按名称解析
                int dim, tag; char name[256];
                if (sscanf(line, "%d %d \"%255[^\"]\"", &dim, &tag, name) == 3)
                    m.name2tag[name] = tag;
            } else if (sec == NODES) {
                long id; double xx, yy, zz;
                if (sscanf(line, "%ld %lf %lf %lf", &id, &xx, &yy, &zz) == 4) {
                    if ((size_t)id > m.x.size()) { m.x.resize(id); m.y.resize(id); }
                    m.x[id - 1] = xx; m.y[id - 1] = yy;
                }
            } else {
                long id; int etype, ntags;
                if (sscanf(line, "%ld %d %d", &id, &etype, &ntags) == 3 && etype == 2) {
                    int t1 = 0, t2 = 0; long a, b, c;
                    sscanf(line, "%ld %d %d %d %d %ld %ld %ld",
                           &id, &etype, &ntags, &t1, &t2, &a, &b, &c);
                    m.tri.push_back({(int)(a - 1), (int)(b - 1), (int)(c - 1), t1});
                }
            }
        }
        fclose(f);
        return m;
    }
    double rmax() const {
        double r = 0;
        for (size_t i = 0; i < x.size(); i++) r = max(r, hypot(x[i], y[i]));
        return r;
    }
    double rmax_phys(int phys) const {   // 该物理域的最大节点半径
        double r = 0;
        for (auto& t : tri) if (t.phys == phys)
            for (int k : t.n) r = max(r, hypot(x[k], y[k]));
        return r;
    }
};

// ---- P1 FEM 求解: A_z 标量磁矢位, Dirichlet r > R-1e-6 → A=0
// nu[i]: 每单元磁阻率; src: 单元源 (可选) f_i += s0[i]*twoA/6 (体电流 J)
//       或 PM: f_i = 0.5*(hx*c_i - hy*b_i)*sgn(twoA)
struct Field {
    Eigen::VectorXd A;
    vector<double> Bx, By;    // 逐单元常数 B
    Msh m;
    vector<double> twoA, cx, cy;   // 单元几何缓存
    Field(const Msh& m_) : m(m_) {}
    void geom() {
        twoA.resize(m.tri.size()); cx.resize(m.tri.size()); cy.resize(m.tri.size());
        for (size_t e = 0; e < m.tri.size(); e++) {
            int i = m.tri[e].n[0], j = m.tri[e].n[1], k = m.tri[e].n[2];
            twoA[e] = (m.x[j] - m.x[i]) * (m.y[k] - m.y[i]) -
                      (m.x[k] - m.x[i]) * (m.y[j] - m.y[i]);   // 有符号 2·面积
            cx[e] = (m.x[i] + m.x[j] + m.x[k]) / 3.0;
            cy[e] = (m.y[i] + m.y[j] + m.y[k]) / 3.0;
        }
    }
    // assemble + solve。nu: phys → 磁阻率 1/(μr·μ0);
    // Jset: phys → 电流密度 Jz (体源 f_i += J·|twoA|/6); Pset: phys → (hx,hy) 磁化
    void solve(const map<int, double>& nu, const map<int, double>& Jset,
               const map<int, pair<double, double>>& Pset) {
        geom();
        size_t N = m.x.size();
        typedef Eigen::SparseMatrix<double> Sp;
        vector<Eigen::Triplet<double>> tp;
        tp.reserve(m.tri.size() * 9);
        Eigen::VectorXd b = Eigen::VectorXd::Zero(N);
        for (size_t e = 0; e < m.tri.size(); e++) {
            int i = m.tri[e].n[0], j = m.tri[e].n[1], k = m.tri[e].n[2];
            double bi = m.y[j] - m.y[k], ci = m.x[k] - m.x[j];
            double bj = m.y[k] - m.y[i], cj = m.x[i] - m.x[k];
            double bk = m.y[i] - m.y[j], ck = m.x[j] - m.x[i];
            double ta = 0.5 * twoA[e];
            double nup = nu.at(m.tri[e].phys);   // 单元磁阻率
            double ke[3][3] = {{bi * bi + ci * ci, bi * bj + ci * cj, bi * bk + ci * ck},
                               {bj * bi + cj * ci, bj * bj + cj * cj, bj * bk + cj * ck},
                               {bk * bi + ck * ci, bk * bj + ck * cj, bk * bk + ck * ck}};
            int ids[3] = {i, j, k};
            for (int a = 0; a < 3; a++)
                for (int d = 0; d < 3; d++)
                    tp.emplace_back(ids[a], ids[d], nup * ke[a][d] / (2.0 * fabs(twoA[e])));
            // 体电流源: ∫N_a·J dA = J·|A|/3 = J·|twoA|/6
            auto itJ = Jset.find(m.tri[e].phys);
            if (itJ != Jset.end()) {
                double s = itJ->second * fabs(twoA[e]) / 6.0;
                b[i] += s; b[j] += s; b[k] += s;
            }
            // PM 源 (与 vw_arbiter 同式): f_a = 0.5*(hx*c_a - hy*b_a)*sgn(twoA)
            auto itP = Pset.find(m.tri[e].phys);
            if (itP != Pset.end()) {
                double hx = itP->second.first, hy = itP->second.second;
                double s = (twoA[e] > 0) ? 0.5 : -0.5;
                b[i] += s * (hx * ci - hy * bi);
                b[j] += s * (hx * cj - hy * bj);
                b[k] += s * (hx * ck - hy * bk);
            }
        }
        Sp K(N, N);
        K.setFromTriplets(tp.begin(), tp.end());
        double R = m.rmax();
        for (size_t i = 0; i < N; i++)
            if (hypot(m.x[i], m.y[i]) > R - 1e-6) {
                for (Sp::InnerIterator it(K, i); it; ++it) it.valueRef() = 0.0;
                K.coeffRef(i, i) = 1.0;
                b[i] = 0.0;
            }
        Eigen::SimplicialLLT<Sp> llt;
        llt.compute(K);
        if (llt.info() != Eigen::Success) { fprintf(stderr, "LLT failed\n"); exit(2); }
        A = llt.solve(b);
        if (A.hasNaN()) { fprintf(stderr, "NaN in solution\n"); exit(2); }
        // 逐单元 B (P0 常数)
        Bx.resize(m.tri.size()); By.resize(m.tri.size());
        for (size_t e = 0; e < m.tri.size(); e++) {
            int i = m.tri[e].n[0], j = m.tri[e].n[1], k = m.tri[e].n[2];
            double ci = m.x[k] - m.x[j], cj = m.x[i] - m.x[k], ck = m.x[j] - m.x[i];
            double bi = m.y[j] - m.y[k], bj = m.y[k] - m.y[i], bk = m.y[i] - m.y[j];
            double ta = 0.5 * twoA[e];
            Bx[e] = (A[i] * ci + A[j] * cj + A[k] * ck) / twoA[e];
            By[e] = -(A[i] * bi + A[j] * bj + A[k] * bk) / twoA[e];
            (void)ta;
        }
    }
    // 点定位 (暴力重心坐标) + B 插值
    bool bfield(double px, double py, double& bx, double& by) const {
        for (size_t e = 0; e < m.tri.size(); e++) {
            int i = m.tri[e].n[0], j = m.tri[e].n[1], k = m.tri[e].n[2];
            double x1 = m.x[i], y1 = m.y[i], x2 = m.x[j], y2 = m.y[j], x3 = m.x[k], y3 = m.y[k];
            double det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
            if (fabs(det) < 1e-18) continue;
            double l1 = ((y2 - y3) * (px - x3) + (x3 - x2) * (py - y3)) / det;
            double l2 = ((y3 - y1) * (px - x3) + (x1 - x3) * (py - y3)) / det;
            double l3 = 1.0 - l1 - l2;
            if (l1 >= -1e-12 && l2 >= -1e-12 && l3 >= -1e-12) {
                bx = Bx[e]; by = By[e];   // P0 常数
                return true;
            }
        }
        return false;
    }
    // 圆周采样: 半径 r, n 点, 返回各角度 (θ=0 起, 逆时针) 的 (Bx,By); 失败点跳过
    void ring_sample(double r, int n, vector<double>& th, vector<double>& bx, vector<double>& by) const {
        th.clear(); bx.clear(); by.clear();
        for (int q = 0; q < n; q++) {
            double t = 2.0 * M_PI * q / n;
            double bxx, byy;
            if (bfield(r * cos(t), r * sin(t), bxx, byy)) {
                th.push_back(t); bx.push_back(bxx); by.push_back(byy);
            }
        }
    }
};

// ---------- A1: 单导体 ----------
static void run_coil(const string& msh, double I) {
    Msh m = Msh::read(msh);
    int tc = m.name2tag.at("conductor"), ta = m.name2tag.at("air");
    double rc = m.rmax_phys(tc);                   // conductor 最大半径
    double J = I / (M_PI * rc * rc);
    Field f(m);
    f.solve({{tc, 1 / MU0}, {ta, 1 / MU0}}, {{tc, J}}, {});
    printf("== A1 单导体: I=%.1f A, rc=%.4f mm, J=%.0f A/m², 单元数=%zu\n",
           I, rc * 1e3, J, m.tri.size());
    printf("  r(mm)   B_fem(T)     B_ana(T)    err%%\n");
    double worst = 0;
    for (double r = 0.01; r <= 0.0801; r += 0.01) {
        vector<double> th, bx, by;
        f.ring_sample(r, 16, th, bx, by);
        double acc = 0; int cnt = 0;
        for (size_t q = 0; q < th.size(); q++) {   // B_θ = -Bx sinθ + By cosθ
            acc += -bx[q] * sin(th[q]) + by[q] * cos(th[q]); cnt++;
        }
        double bf = cnt ? acc / cnt : 0.0;
        double ba = MU0 * I / (2 * M_PI * r);
        double err = 100.0 * (bf - ba) / ba;
        worst = max(worst, fabs(err));
        printf("  %5.1f  %10.6f  %10.6f  %+6.3f\n", r * 1e3, bf, ba, err);
    }
    printf("  结论: 最大误差 |err|=%.3f%%  (阶梯 A1 %s)\n",
           worst, worst < 1.0 ? "PASS (<1%)" : "FAIL");
}

// ---------- A2: 均匀横向磁化圆柱 ----------
static void run_cyl(const string& msh, double M) {
    Msh m = Msh::read(msh);
    int tp_ = m.name2tag.at("pm"), ta = m.name2tag.at("air");
    double b = m.rmax_phys(tp_);   // pm 域外半径
    double R = m.rmax();           // Dirichlet 半径
    Field f(m);
    f.solve({{tp_, 1 / MU0}, {ta, 1 / MU0}}, {}, {{tp_, {M, 0.0}}});
    double A4 = M * b * b / 2.0;      // 标势系数 (H 场量纲); B 场量级 μ0·A4
    double Bin = MU0 * M * (1 - (b / R) * (b / R)) / 2.0;
    printf("== A2 磁化圆柱: M=%.4ge5 A/m, b=%.1f mm, R=%.1f mm (R/b=%.0f), 单元数=%zu\n",
           M / 1e5, b * 1e3, R * 1e3, R / b, m.tri.size());
    printf("  解析: B_in = mu0*M*(1-b²/R²)/2 = %.6f T (R→∞ 特例 %.6f = mu0*M/2)\n",
           Bin, MU0 * M / 2);
    vector<double> th, bx, by;
    printf("  [内域 r<b]  r(mm)   Bx_fem(T)   Bx_ana(T)   err%%\n");
    double worst = 0;
    for (double r = 0.005; r <= 0.0451; r += 0.005) {
        f.ring_sample(r, 16, th, bx, by);
        double acc = 0; int cnt = 0;
        for (size_t q = 0; q < th.size(); q++) { acc += bx[q]; cnt++; }
        double bf = cnt ? acc / cnt : 0.0;
        double err = 100.0 * (bf - Bin) / Bin;
        worst = max(worst, fabs(err));
        printf("  %14.1f  %10.6f  %10.6f  %+6.3f\n", r * 1e3, bf, Bin, err);
    }
    // 外域: B_r = μ0·A4·(1/r²+1/R²)cosθ, B_θ = μ0·A4·(1/r²−1/R²)sinθ
    // P0 常数采样噪声大 → 圆周 32 点 |B| 平均, 解析侧取同口径平均
    printf("  [外域 r>b]  r(mm)   |B|_fem(32pt)  |B|_ana(32pt)  err%%\n");
    for (double r = 0.055; r <= 0.1401; r += 0.005) {
        f.ring_sample(r, 32, th, bx, by);
        double acc = 0; int cnt = 0;
        for (size_t q = 0; q < th.size(); q++) {
            acc += hypot(bx[q], by[q]); cnt++;
        }
        double bf = cnt ? acc / cnt : 0.0;
        double ba_acc = 0; int bc = 0;
        for (int q = 0; q < 32; q++) {
            double t = 2 * M_PI * q / 32;
            double br = MU0 * A4 * (1 / (r * r) + 1 / (R * R)) * cos(t);
            double bt = MU0 * A4 * (1 / (r * r) - 1 / (R * R)) * sin(t);
            ba_acc += hypot(br, bt); bc++;
        }
        double ba = ba_acc / bc;
        double err = fabs(ba) > 1e-15 ? 100.0 * (bf - ba) / ba : 0.0;
        worst = max(worst, fabs(err));
        printf("  %14.1f  %12.6f  %14.6f  %+6.3f\n", r * 1e3, bf, ba, err);
    }
    printf("  结论: 最大误差 |err|=%.3f%%  (阶梯 A2 %s)\n",
           worst, worst < 3.0 ? "PASS (<3%)" : "FAIL");
}

// ---------- B: 无载气隙谱 + Carter 1D ----------
static void run_motor(const string& msh, const string& pos) {
    Msh m = Msh::read(msh);
    map<int, double> mur = {{1, 1.0}, {2, 7000.0}, {7, 7000.0}, {3, 1.05}, {4, 1.05}};
    // 读 pos (VT(9坐标){9值}) → 质心哈希 → 用 FEM 单元质心匹配取 B (外部位场)
    FILE* fp = fopen(pos.c_str(), "r");
    if (!fp) { fprintf(stderr, "cannot open %s\n", pos.c_str()); exit(1); }
    char line[8192];
    map<pair<long long, long long>, pair<double, double>> bmap;
    while (fgets(line, sizeof line, fp)) {
        double c[9], v[9];
        char* p1 = strchr(line, '(');
        char* p2 = strchr(line, ')');
        char* p3 = p2 ? strchr(p2, '{') : nullptr;
        char* p4 = p3 ? strchr(p3, '}') : nullptr;
        if (!p1 || !p2 || !p3 || !p4) continue;
        if (sscanf(p1 + 1, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                   &c[0], &c[1], &c[2], &c[3], &c[4], &c[5], &c[6], &c[7], &c[8]) != 9) continue;
        if (sscanf(p3 + 1, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                   &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6], &v[7], &v[8]) != 9) continue;
        double cx = (c[0] + c[3] + c[6]) / 3.0, cy = (c[1] + c[4] + c[7]) / 3.0;
        bmap[{llround(cx * 1e9), llround(cy * 1e9)}] = {v[0], v[1]};   // 顶点0 的 B (P0×3)
    }
    fclose(fp);
    // 单元几何 + 质心
    vector<double> twoA(m.tri.size());
    vector<array<double, 2>> cen(m.tri.size());
    int matched = 0;
    vector<double> bex(m.tri.size()), bey(m.tri.size());
    for (size_t e = 0; e < m.tri.size(); e++) {
        int i = m.tri[e].n[0], j = m.tri[e].n[1], k = m.tri[e].n[2];
        twoA[e] = (m.x[j] - m.x[i]) * (m.y[k] - m.y[i]) - (m.x[k] - m.x[i]) * (m.y[j] - m.y[i]);
        cen[e] = {(m.x[i] + m.x[j] + m.x[k]) / 3, (m.y[i] + m.y[j] + m.y[k]) / 3};
        auto it = bmap.find({llround(cen[e][0] * 1e9), llround(cen[e][1] * 1e9)});
        if (it != bmap.end()) { bex[e] = it->second.first; bey[e] = it->second.second; matched++; }
    }
    printf("== B 气隙谱: pos 单元匹配 %d/%zu\n", matched, m.tri.size());
    // r=29.5mm 圆周 720 点: 就近单元 (质心距离最小) 的 B → 法向分量
    const double rg = 0.0295;
    const int NS = 720;
    vector<double> bn(NS);
    for (int q = 0; q < NS; q++) {
        double t = 2.0 * M_PI * q / NS;
        double px = rg * cos(t), py = rg * sin(t);
        int best = -1; double bd = 1e18;
        for (size_t e = 0; e < m.tri.size(); e++) {
            double dx = cen[e][0] - px, dy = cen[e][1] - py;
            double d = dx * dx + dy * dy;
            if (d < bd) { bd = d; best = (int)e; }
        }
        if (best < 0) { fprintf(stderr, "gap sample failed\n"); exit(3); }
        bn[q] = bex[best] * cos(t) + bey[best] * sin(t);
    }
    // Fourier: Bk 幅值 (k=1..24)
    double mean = 0; for (double v : bn) mean += v; mean /= NS;
    double B[25] = {0};
    for (int k = 1; k <= 24; k++) {
        double c = 0, s = 0;
        for (int q = 0; q < NS; q++) {
            double t = 2.0 * M_PI * q / NS;
            c += bn[q] * cos(k * t); s += bn[q] * sin(k * t);
        }
        B[k] = 2.0 / NS * hypot(c, s);
    }
    double thd = 0; for (int k = 1; k <= 24; k++) if (k != 2) thd += B[k] * B[k];
    thd = 100.0 * sqrt(thd) / B[2];
    // 口径对齐驱动 run_emag_getdp.py:204: pole_mean = Bn[d0 < mag_half_deg].mean()
    // 磁极半角 33.75° (极弧系数 0.75 × 极距 90°), 非 |Bn|
    double pole = 0; int np = 0;
    for (int q = 0; q < NS; q++) {
        double t = 2.0 * M_PI * q / NS;
        double dd = fabs(atan2(sin(t), cos(t)));
        if (dd < 33.75 * M_PI / 180) { pole += bn[q]; np++; }
    }
    pole /= np;
    printf("  极均 Bn(|θ|<33.75°) = %.4f T   (驱动/GetDP 参考 0.7585)\n", pole);
    printf("  Bg1(k=2) = %.4f T   (驱动 0.9155)\n", B[2]);
    printf("  槽谐波 k=12 = %.4f T, k=24 = %.4f T\n", B[12], B[24]);
    printf("  THD(k!=2,<=24) = %.1f%%   (驱动 21.6%%)\n", thd);
    // Carter 1D 磁路
    double g = 1.0, hm = 3.0, Br = 1.16, mur_iron = 1.05;   // mm, T
    double b0 = 2 * 30.0 * sin(2.0 * M_PI / 180.0);        // 槽开口宽 (R_si=30mm, 半角 2°)
    double ts = 2 * M_PI * 30.0 / 12;                      // 槽距
    double gam = (b0 / g) * (b0 / g) / (5 + b0 / g);
    double kc = ts / (ts - gam * g);
    double bgc = Br * hm / (hm + mur_iron * kc * g);
    double bgn = Br * hm / (hm + mur_iron * g);
    printf("  Carter: b0=%.3f mm, ts=%.3f mm, γ=%.4f, k_c=%.4f\n", b0, ts, gam, kc);
    printf("  1D 磁路: Bg(no Carter)=%.4f T, Bg(Carter)=%.4f T\n", bgn, bgc);
    printf("  交叉: FEM 极均/1D(Carter) = %.3f, FEM Bg1/1D(no Carter) = %.3f\n", pole / bgc, B[2] / bgn);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "用法:\n  %s coil <msh> <I_A>\n  %s cyl <msh> <M_A/m>\n  %s motor <msh> <pos>\n",
                argv[0], argv[0], argv[0]);
        return 1;
    }
    string mode = argv[1];
    if (mode == "coil" && argc >= 4) run_coil(argv[2], atof(argv[3]));
    else if (mode == "cyl" && argc >= 4) run_cyl(argv[2], atof(argv[3]));
    else if (mode == "motor" && argc >= 4) run_motor(argv[2], argv[3]);
    else { fprintf(stderr, "未知模式\n"); return 1; }
    return 0;
}
