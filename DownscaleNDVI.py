import pandas as pd
import numpy as np
import rasterio
from rasterio.plot import show
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import os

# ======= Đường dẫn =======
csv_path = r"D:\Downscale NDVI\XGBoost_Training_Texture_ThanhHoa.csv"
modis_path = r"D:\Downscale NDVI\NDVI_MODIS.tif"
predictor_tif = r"D:\Downscale NDVI\ThanhHoABoostInputs.tif"
output_predicted_tif = r"D:\Downscale NDVI\NDVI_XGBoost_Predicted.tif"

# ======= Load CSV và xử lý =======
df = pd.read_csv(csv_path)

# Loại bỏ các cột không quan trọng
cols_to_drop = ['system:index', '.geo', 'Unnamed: 0'] + \
               [col for col in df.columns if any(kw in col for kw in ['_savg', '_asm', '_corr', '_dent', '_shade', '_idm', '_contrast'])]
df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

# ======= Tách tập train/test =======
X = df.drop(columns=["NDVI_Landsat"])
y = df["NDVI_Landsat"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ======= Huấn luyện mô hình XGBoost =======
model = xgb.XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1, n_jobs=-1)
model.fit(X_train, y_train)

# ======= Đánh giá mô hình =======
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"🎯 RMSE = {rmse:.4f}, R² = {r2:.4f}")

# ======= Vẽ biểu đồ kết quả =======
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(y_test, bins=30, kde=True, color='green', label='Thực tế')
sns.histplot(y_pred, bins=30, kde=True, color='orange', label='Dự đoán')
plt.legend()
plt.title("Histogram NDVI (Test set)")
plt.xlabel("NDVI_Landsat")
plt.ylabel("Count")

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.5, edgecolor='k')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlabel("NDVI thực tế")
plt.ylabel("NDVI dự đoán")
plt.title("Scatter plot: NDVI thực tế vs dự đoán")

plt.tight_layout()
plt.show()

# ======= Tên 60 bands theo thứ tự ảnh raster (FULL) =======
all_band_names = [
    'NDVI_MODIS', 'EVI', 'SAVI', 'NBR', 'DEM', 'Slope',
    'EVI_asm', 'EVI_contrast', 'EVI_corr', 'EVI_var', 'EVI_idm', 'EVI_savg', 'EVI_svar', 'EVI_sent', 'EVI_ent', 'EVI_dvar', 'EVI_dent',
    'EVI_imcorr1', 'EVI_imcorr2', 'EVI_maxcorr', 'EVI_diss', 'EVI_inertia', 'EVI_shade', 'EVI_prom',
    'SAVI_asm', 'SAVI_contrast', 'SAVI_corr', 'SAVI_var', 'SAVI_idm', 'SAVI_savg', 'SAVI_svar', 'SAVI_sent', 'SAVI_ent', 'SAVI_dvar', 'SAVI_dent',
    'SAVI_imcorr1', 'SAVI_imcorr2', 'SAVI_maxcorr', 'SAVI_diss', 'SAVI_inertia', 'SAVI_shade', 'SAVI_prom',
    'NBR_asm', 'NBR_contrast', 'NBR_corr', 'NBR_var', 'NBR_idm', 'NBR_savg', 'NBR_svar', 'NBR_sent', 'NBR_ent', 'NBR_dvar', 'NBR_dent',
    'NBR_imcorr1', 'NBR_imcorr2', 'NBR_maxcorr', 'NBR_diss', 'NBR_inertia', 'NBR_shade', 'NBR_prom'
]

# ======= Apply model cho ảnh raster =======
with rasterio.open(predictor_tif) as src:
    profile = src.profile
    profile.update(dtype='float32', count=1)
    
    predictors_array = src.read().astype('float32')  # shape: (bands, height, width)
    n_bands, height, width = predictors_array.shape

    # Chuyển về 2D (pixels, bands)
    X_pred = predictors_array.reshape(n_bands, -1).T

    # Mapping tên theo all_band_names
    df_pred = pd.DataFrame(X_pred, columns=all_band_names)

    # Chỉ giữ lại các cột đã dùng trong huấn luyện
    df_pred = df_pred[X.columns.tolist()]

    # Xử lý NaN hoặc vô cực
    df_pred = df_pred.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Dự đoán NDVI
    y_predicted = model.predict(df_pred).reshape(height, width)

# ======= Lưu ảnh NDVI dự đoán =======
with rasterio.open(output_predicted_tif, 'w', **profile) as dst:
    dst.write(y_predicted.astype('float32'), 1)

print("✅ NDVI dự đoán đã lưu tại:", output_predicted_tif)

# ======= So sánh ảnh NDVI MODIS với ảnh dự đoán =======
with rasterio.open(modis_path) as src:
    ndvi_modis = src.read(1).astype('float32')

# Vẽ ảnh MODIS và ảnh dự đoán
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].imshow(ndvi_modis, cmap='YlGn', vmin=0, vmax=1)
axs[0].set_title("NDVI MODIS (Base)")
axs[0].axis('off')

axs[1].imshow(y_predicted, cmap='YlGn', vmin=0, vmax=1)
axs[1].set_title("NDVI Dự đoán từ XGBoost")
axs[1].axis('off')

plt.tight_layout()
plt.show()
