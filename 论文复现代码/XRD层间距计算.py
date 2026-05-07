import numpy as np
import pandas as pd

n = 1
lambda_nm = 0.15406  # Cu Kα 波长 (nm)

def calculate_d001(theta_2):
    theta = np.radians(theta_2 / 2)
    d = (n * lambda_nm) / (2 * np.sin(theta))
    return d

# ========== 读取Excel（不管表头，直接读） ==========
df = pd.read_excel("XRD导电角数据.xlsx", sheet_name="Sheet1", header=1)

# ========== 终极方案：按「列的位置」取数据，不用列名！ ==========
# 第2列（索引1）= 生物分子，第3列（索引2）= 加载量Q，第5列（索引4）= 2θ (°)
# 筛选：第2列的值 == "赖氨酸"
lys_df = df[df.iloc[:, 1] == "赖氨酸"].copy()

# 取第3列（索引2）= 加载量Q
Q_lys = lys_df.iloc[:, 2].tolist()
# 取第5列（索引4）= 2θ (°)
theta2_lys = lys_df.iloc[:, 4].tolist()

# 计算层间距
d001_lys = [round(calculate_d001(t), 3) for t in theta2_lys]

# 构建结果
xrd_df = pd.DataFrame({
    "Lys加载量Q (μmol/g)": Q_lys,
    "2θ (°)": theta2_lys,
    "d001 (nm)": d001_lys
})

# 保存
print("\n✅ 计算结果：")
print(xrd_df)
xrd_df.to_excel("TableS2_XRD层间距计算.xlsx", index=False)
print("\n💾 结果已保存为：TableS2_XRD层间距计算.xlsx")