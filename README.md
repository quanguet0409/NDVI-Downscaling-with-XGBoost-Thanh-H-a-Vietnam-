🛰️ NDVI Downscaling with XGBoost (Thanh Hóa, Vietnam)
This repository demonstrates a machine learning-based approach to downscale MODIS NDVI to Landsat resolution using XGBoost. The implementation is adapted from:

Robinson et al. (2019) - Fusing MODIS with Landsat 8 data to downscale weekly normalized difference vegetation index estimates for central Great Basin rangelands, USA

📍 Study Area
Location: Thanh Hóa Province, Vietnam

MODIS NDVI resolution: 250m

Target NDVI resolution: 30m (Landsat scale)

📂 Input Data

Source	Description
MODIS	NDVI weekly composite (low resolution)
Landsat 8	Spectral indices: EVI, SAVI, NBR
Terrain	DEM and Slope
Texture	GLCM features of EVI/SAVI/NBR (entropy, variance, etc.)
Training CSV	5,000 sample points extracted via Google Earth Engine
All raster layers were exported from Google Earth Engine and used as model inputs.

🧠 Methodology (Python workflow)
🧪 1. Prepare training data
Load .csv exported from GEE (XGBoost_Training_Texture_ThanhHoa.csv)

Remove unnecessary columns and filter useful features

Split into training and testing sets (80/20)

🤖 2. Train XGBoost model
Use xgboost.XGBRegressor with optimized parameters (max_depth, n_estimators)

Evaluate with RMSE and R²

📉 3. Visualize model performance
Histogram of predicted vs actual NDVI

Scatter plot of test prediction

🗺️ 4. Predict NDVI from raster inputs
Load full-size raster input (ThanhHoa_XGBoostInputs.tif)

Reshape bands to (pixels, features)

Predict NDVI and reshape to original raster size

💾 5. Export predicted NDVI
Save output NDVI to GeoTIFF (NDVI_XGBoost_Predicted.tif) using rasterio

📊 6. Compare MODIS vs Predicted NDVI
Load original MODIS NDVI (NDVI_MODIS.tif)

Plot both rasters side-by-side using matplotlib

🛠️ Technologies
Python, XGBoost, Pandas, Scikit-learn

Rasterio: Raster data reading/writing

Matplotlib, Seaborn: Visualization


