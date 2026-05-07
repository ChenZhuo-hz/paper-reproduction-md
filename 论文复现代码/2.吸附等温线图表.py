import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# 设置图表样式（匹配论文期刊风格）
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# ===================== 拟合函数（简化版，直接返回斜率和拟合线）=====================
def linear_fit(x, y):
    slope, intercept, r2, p, se = stats.linregress(x, y)
    y_fit = slope * x + intercept
    return slope, intercept, r2, y_fit

# ===================== 数据准备（复现Fig S1 Na-MMT单分子吸附：Lys/His/Thr）=====================
# 平衡浓度Ceq（μmol/L）
Ceq = np.array([125, 250, 500, 1000, 2000, 4000])
# 各分子吸附量Q（μmol/g），匹配论文Kd趋势
Q_lys = 0.0164 * Ceq  # Lys Kd=16.43
Q_his = 0.0034 * Ceq  # His Kd=3.42
Q_thr = 0.0020 * Ceq  # Thr Kd=1.97

# 线性拟合
slope_lys, _, r2_lys, y_lys = linear_fit(Ceq, Q_lys)
slope_his, _, r2_his, y_his = linear_fit(Ceq, Q_his)
slope_thr, _, r2_thr, y_thr = linear_fit(Ceq, Q_thr)

# ===================== 绘制图表（论文同款：散点+虚线拟合）=====================
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
# 绘制散点和拟合线
ax.scatter(Ceq, Q_lys, c="#e74c3c", s=50, label=f'Lys (R²={r2_lys:.4f})')
ax.plot(Ceq, y_lys, "--", c="#e74c3c", linewidth=2)
ax.scatter(Ceq, Q_his, c="#3498db", s=50, label=f'His (R²={r2_his:.4f})')
ax.plot(Ceq, y_his, "--", c="#3498db", linewidth=2)
ax.scatter(Ceq, Q_thr, c="#2ecc71", s=50, label=f'Thr (R²={r2_thr:.4f})')
ax.plot(Ceq, y_thr, "--", c="#2ecc71", linewidth=2)

# 设置坐标轴和标签（匹配论文）
ax.set_xlabel('$C_{eq}$ (μmol/L)', fontsize=14, fontweight='bold')
ax.set_ylabel('Q (μmol compound $g^{-1}$ clay)', fontsize=14, fontweight='bold')
ax.set_title('Adsorption isotherm on Na-MMT (Fig S1 Reproduction)', fontsize=16, fontweight='bold', pad=20)
ax.tick_params(axis='both', which='major', labelsize=12)
# 图例
ax.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
# 保存图表（高清，可直接用于论文复现）
plt.tight_layout()
plt.savefig("FigS1_Na-MMT吸附等温线.png", dpi=300, bbox_inches='tight')
plt.close()

print("图表已保存为：FigS1_Na-MMT吸附等温线.png")
# ===================== 拓展：Mg-MMT/混合分子吸附可直接替换Q数据重复上述步骤 =====================