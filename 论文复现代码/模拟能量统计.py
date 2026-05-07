import numpy as np
import pandas as pd

n = 1  # 一级衍射
lambda_nm = 0.15406  # Cu Kα 射线波长 (nm)

def calculate_d001(theta_2):
    theta = np.radians(theta_2 / 2)
    d = (n * lambda_nm) / (2 * np.sin(theta))
    return d

# ========== 读取Excel（容错表头） ==========
# 先尝试用第2行作为表头，如果不行就不设header，自动识别
try:
    df = pd.read_excel("XRD导电角数据.xlsx", sheet_name="Sheet1", header=1)
except:
    df = pd.read_excel("XRD导电角数据.xlsx", sheet_name="Sheet1")

print("✅ Excel 列名：", df.columns.tolist())

# ========== 模糊筛选赖氨酸（容错空格/换行） ==========
# 找「生物分子」相关列（不管列名带不带空格）
bio_col = None
for col in df.columns:
    if "生物分子" in str(col):
        bio_col = col
        break

# 筛选：列值包含「赖氨酸」（容错前后空格）
lys_df = df[df[bio_col].astype(str).str.contains("赖氨酸", na=False)].copy()

print(f"\n🔍 筛选到 {len(lys_df)} 行赖氨酸数据")

# ========== 提取数据（容错列名） ==========
# 找「加载量Q」列
q_col = None
for col in df.columns:
    if "加载量Q" in str(col):
        q_col = col
        break
Q_lys = lys_df[q_col].tolist()

# 找「2θ」列
theta2_col = None
for col in df.columns:
    if "2θ" in str(col):
        theta2_col = col
        break
theta2_lys = lys_df[theta2_col].tolist()

# 计算层间距
d001_lys = [round(calculate_d001(t), 3) for t in theta2_lys]

# 构建结果
xrd_df = pd.DataFrame({
    "Lys加载量Q (μmol/g)": Q_lys,
    "2θ (°)": theta2_lys,
    "d001 (nm)": d001_lys
})

# 保存
print("\n📊 计算结果：")
print(xrd_df)
xrd_df.to_excel("TableS2_XRD层间距计算.xlsx", index=False)
print("\n💾 结果已保存为：TableS2_XRD层间距计算.xlsx")