import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.interpolate import griddata

# ============================================================
# 1. Зчитування даних сенсорного слайдера
# ============================================================
file_path = 'C:/Слайдер_13_11_25/data_slider_1.xlsx'
sheet_name = 'data'

df = pd.read_excel(file_path, sheet_name=sheet_name)

# 1-й стовпець -> X, 2-й стовпець -> Y,
# решта 8 стовпців -> сигнали сенсорів S1...S8
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values
S = df.iloc[:, 2:].values

# ============================================================
# 2. Просторова карта сигналу окремого сенсора
# ============================================================
def plot_sensor_map(sensor_id=0):
    grid_x, grid_y = np.mgrid[min(x):max(x):100j, min(y):max(y):100j]
    grid_z = griddata((x, y), S[:, sensor_id], (grid_x, grid_y), method='cubic')

    plt.figure(figsize=(8, 5))
    plt.title(f'Spatial map of sensor S{sensor_id + 1}')
    plt.imshow(grid_z.T, extent=(min(x), max(x), min(y), max(y)),
               origin='lower', aspect='auto')
    plt.colorbar(label='Signal')
    plt.xlabel('X, mm')
    plt.ylabel('Y, mm')
    plt.tight_layout()
    plt.show()

# ============================================================
# 3. Карта домінуючого сенсора
# ============================================================
def plot_dominant_sensor_map():
    labels = np.argmax(S, axis=1)

    plt.figure(figsize=(8, 5))
    plt.title('Dominant sensor map')
    scatter = plt.scatter(x, y, c=labels, s=14)
    plt.colorbar(scatter, label='Sensor ID')
    plt.xlabel('X, mm')
    plt.ylabel('Y, mm')
    plt.tight_layout()
    plt.show()

# ============================================================
# 4. PCA-представлення восьми сенсорних сигналів
# ============================================================
def plot_pca():
    pca = PCA(n_components=2)
    S_pca = pca.fit_transform(S)

    plt.figure(figsize=(7, 6))
    plt.title('PCA of sensor signals')
    scatter = plt.scatter(S_pca[:, 0], S_pca[:, 1], c=x, s=10)
    plt.colorbar(scatter, label='X coordinate, mm')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.show()

    print('Explained variance ratio:', pca.explained_variance_ratio_)

# ============================================================
# 5. Кореляційна матриця сенсорних сигналів
# ============================================================
def plot_correlation_matrix():
    corr = np.corrcoef(S.T)

    plt.figure(figsize=(7, 6))
    plt.title('Sensor correlation matrix')
    sns.heatmap(corr, annot=True, fmt='.2f', square=True)
    plt.xlabel('Sensors')
    plt.ylabel('Sensors')
    plt.tight_layout()
    plt.show()

# ============================================================
# 6. Аналіз похибок прогнозування координати
# ============================================================
def plot_error_analysis(error_file):
    err_df = pd.read_excel(error_file)

    x_true = err_df.iloc[:, 0].values
    y_true = err_df.iloc[:, 1].values
    error = err_df.iloc[:, -1].values

    plt.figure(figsize=(8, 4))
    plt.title('Prediction error by X coordinate')
    plt.scatter(x_true, error, s=10)
    plt.xlabel('X coordinate, mm')
    plt.ylabel('Error, mm')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 4))
    plt.title('Error histogram')
    plt.hist(error, bins=30)
    plt.xlabel('Error, mm')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    grid_x, grid_y = np.mgrid[min(x_true):max(x_true):100j,
                              min(y_true):max(y_true):100j]
    grid_err = griddata((x_true, y_true), error, (grid_x, grid_y), method='cubic')

    plt.figure(figsize=(9, 5))
    plt.title('Spatial error map')
    plt.imshow(grid_err.T, extent=(min(x_true), max(x_true), min(y_true), max(y_true)),
               origin='lower', aspect='auto')
    plt.colorbar(label='Error, mm')
    plt.xlabel('X coordinate, mm')
    plt.ylabel('Y coordinate, mm')
    plt.tight_layout()
    plt.show()

# ============================================================
# 7. Запуск основних побудов
# ============================================================
plot_sensor_map(sensor_id=0)
plot_sensor_map(sensor_id=1)
plot_sensor_map(sensor_id=5)
plot_dominant_sensor_map()
plot_pca()
plot_correlation_matrix()
# plot_error_analysis('C:/Слайдер_13_11_25/Err_data_slider_1.xlsx')
