# Documentación del código – DengAI: Predicción de la propagación del dengue

Este documento explica el código del proyecto y para qué sirve cada parte, e incluye huecos para insertar **capturas de las gráficas** generadas al ejecutar el notebook.

---

## Índice del documento

1. [Importación del dataset](#1-importación-del-dataset)
2. [Preparación y calidad de los datos](#2-preparación-y-calidad-de-los-datos)
3. [Selección de características](#3-selección-de-características)
4. [División Train / Validación / Test](#4-división-train--validación--test)
5. [Modelo 1: Naive Bayes](#5-modelo-1-naive-bayes)
6. [Modelo 2: KNN](#6-modelo-2-knn)
7. [Modelo 3: Random Forest](#7-modelo-3-random-forest)
8. [Comparativa de modelos (gráficos)](#8-comparativa-de-modelos-gráficos)
9. [Predicción y validación](#9-predicción-y-validación)
10. [Generación del fichero de submit](#10-generación-del-fichero-de-submit)

---

## 1. Importación del dataset

### Para qué sirve

Cargar los tres ficheros del proyecto (etiquetas de entrenamiento, características de entrenamiento y características de test) desde una **carpeta local**, desde **Google Drive** (Colab) o desde una **URL de GitHub** (raw), para cumplir el criterio de usar Drive o GitHub como origen.

### Código explicado

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Tamaño por defecto de las figuras y estilo de seaborn
plt.rcParams['figure.figsize'] = (10, 6)
sns.set_style('whitegrid')

def load_data_from_github(base_url=None):
    """Carga los CSV desde una URL base de GitHub (enlace raw)."""
    if base_url is None:
        return None
    try:
        labels = pd.read_csv(f"{base_url}/dengue_labels_train.csv")
        feat_train = pd.read_csv(f"{base_url}/dengue_features_train.csv")
        feat_test = pd.read_csv(f"{base_url}/dengue_features_test.csv")
        return labels, feat_train, feat_test
    except Exception as e:
        print(f"Error cargando desde GitHub: {e}")
        return None

def load_data_local(data_dir='data'):
    """Carga los CSV desde una carpeta local (p. ej. data/ o ruta en Drive)."""
    data_path = Path(data_dir)
    labels = pd.read_csv(data_path / 'dengue_labels_train.csv')
    feat_train = pd.read_csv(data_path / 'dengue_features_train.csv')
    feat_test = pd.read_csv(data_path / 'dengue_features_test.csv')
    return labels, feat_train, feat_test

# Uso: carpeta local (cambiar a ruta de Drive o usar load_data_from_github con URL)
DATA_DIR = 'data'
labels_train, features_train, features_test = load_data_local(DATA_DIR)

print("Labels train:", labels_train.shape)
print("Features train:", features_train.shape)
print("Features test:", features_test.shape)
labels_train.head()
```

- **`labels_train`**: filas por (city, year, weekofyear) con la columna **total_cases** (objetivo a predecir).
- **`features_train`**: mismas filas con variables climáticas y de satélite (NDVI, precipitación, temperaturas, etc.).
- **`features_test`**: mismas columnas que features_train pero para las semanas/ciudades donde hay que predecir total_cases.

### Captura (opcional)

Si quieres incluir una captura de la salida de `labels_train.head()` o de las dimensiones impresas:

<!-- INSERTAR CAPTURA: Salida de importación (shape y head de labels_train) -->
![Salida importación](capturas/01_importacion.png)

---

## 2. Preparación y calidad de los datos

### Para qué sirve

- Unir características y etiquetas en un solo DataFrame.
- Eliminar la columna de fecha (no numérica).
- Codificar la ciudad (sj → 1, iq → 0).
- Tratar valores faltantes (mediana) y **normalizar** con `StandardScaler` para que los modelos trabajen en una escala común.

### Código explicado (unión, limpieza y relleno de faltantes)

```python
# Unir por city, year, weekofyear para tener X e y en la misma tabla
df = features_train.merge(labels_train, on=['city', 'year', 'weekofyear'], how='inner')

# Quitar la columna de fecha (no se usa como predictor numérico)
col_fecha = 'week_start_date'
if col_fecha in df.columns:
    df = df.drop(columns=[col_fecha])
if col_fecha in features_test.columns:
    features_test_clean = features_test.drop(columns=[col_fecha]).copy()
else:
    features_test_clean = features_test.copy()

# Ciudad como 0/1: San Juan = 1, Iquitos = 0
df['city'] = (df['city'] == 'sj').astype(int)
features_test_clean['city'] = (features_test_clean['city'] == 'sj').astype(int)

# Asegurar que todas las columnas numéricas son tipo float (por si hay cadenas vacías)
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = pd.to_numeric(df[c], errors='coerce')
for c in features_test_clean.select_dtypes(include=[np.number]).columns:
    features_test_clean[c] = pd.to_numeric(features_test_clean[c], errors='coerce')

# Rellenar NaN con la mediana de cada columna (robusto a outliers)
target_col = 'total_cases'
feature_cols = [c for c in df.columns if c != target_col]
medians = df[feature_cols].median()
df[feature_cols] = df[feature_cols].fillna(medians)
features_test_clean[feature_cols] = features_test_clean[feature_cols].fillna(medians)
```

### Código explicado (normalización)

```python
from sklearn.preprocessing import StandardScaler

X_full = df[feature_cols].copy()
y_full = df[target_col].copy()

# StandardScaler: media 0 y varianza 1 por columna (mejor para KNN y NB)
scaler = StandardScaler()
X_full_norm = pd.DataFrame(
    scaler.fit_transform(X_full),
    columns=feature_cols,
    index=X_full.index
)
# El test se transforma con la misma media y std del train (fit solo en train)
X_test_norm = pd.DataFrame(
    scaler.transform(features_test_clean[feature_cols]),
    columns=feature_cols,
    index=features_test_clean.index
)
```

### Captura (opcional)

<!-- INSERTAR CAPTURA: Valores faltantes o describe() tras preparación -->
![Preparación datos](capturas/02_preparacion.png)

---

## 3. Selección de características

### Para qué sirve

Usar **métodos y gráficos** para elegir qué variables predictoras usar: correlación con `total_cases`, importancia con un árbol de decisión y matriz de correlación entre predictores para ver redundancia.

### Código explicado (correlación con el target)

```python
# Correlación de cada predictor con total_cases (Pearson)
corr_target = X_full_norm.copy()
corr_target['total_cases'] = y_full
correlations = corr_target.corr()['total_cases'].drop('total_cases').sort_values(key=abs, ascending=False)

# Gráfico de barras horizontales: cuánto se relaciona cada variable con el objetivo
fig, ax = plt.subplots(figsize=(10, 8))
correlations.plot(kind='barh', ax=ax, color='steelblue', edgecolor='navy')
ax.set_title('Selección de características: correlación con total_cases')
ax.set_xlabel('Correlación (Pearson)')
plt.tight_layout()
plt.show()
```

### Hueco para captura – Gráfico de correlación con total_cases

<!-- INSERTAR CAPTURA: Gráfico de barras de correlación con total_cases -->
![Correlación con total_cases](capturas/03_correlacion_target.png)

---

### Código explicado (importancia con árbol de decisión)

```python
from sklearn.tree import DecisionTreeRegressor

# Un árbol rápido para obtener importancia de cada variable (herramienta gráfica de selección)
dt = DecisionTreeRegressor(random_state=42, max_depth=10)
dt.fit(X_full_norm, y_full)
imp = pd.Series(dt.feature_importances_, index=feature_cols).sort_values(ascending=False)

fig, ax = plt.subplots(figsize=(10, 8))
imp.plot(kind='barh', ax=ax, color='darkgreen', alpha=0.8)
ax.set_title('Selección de características: importancia (Decision Tree)')
ax.set_xlabel('Importancia')
plt.tight_layout()
plt.show()
```

### Hueco para captura – Gráfico de importancia (árbol)

<!-- INSERTAR CAPTURA: Gráfico de importancia de características (Decision Tree) -->
![Importancia características](capturas/04_importancia_arbol.png)

---

### Código explicado (matriz de correlación y selección final)

```python
# Matriz de correlación entre todas las variables (detectar redundancia)
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(X_full_norm.corr(), cmap='RdBu_r', center=0, ax=ax, square=True, linewidths=0.5, fmt='.1f')
ax.set_title('Matriz de correlación entre características (selección de redundantes)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Criterio: quedarnos con variables correlacionadas con el target o con importancia alta
umbral_corr = 0.05
selected_by_corr = correlations[abs(correlations) >= umbral_corr].index.tolist()
selected_by_imp = imp[imp >= imp.quantile(0.2)].index.tolist()
selected_features = list(dict.fromkeys(selected_by_corr + selected_by_imp))
if not selected_features:
    selected_features = feature_cols
print("Características seleccionadas:", selected_features)
```

### Hueco para captura – Matriz de correlación

<!-- INSERTAR CAPTURA: Heatmap de correlación entre características -->
![Matriz correlación](capturas/05_matriz_correlacion.png)

---

## 4. División Train / Validación / Test

### Para qué sirve

Tener tres conjuntos: **train** para entrenar, **validación** para elegir hiperparámetros y comparar modelos, y **test** para una evaluación final sin haber tocado esos datos durante el ajuste.

### Código explicado

```python
from sklearn.model_selection import train_test_split

X = X_full_norm[selected_features].copy()
y = y_full.copy()

# 70% train, 15% validación, 15% test (sobre el 30% restante se hace 50%-50%)
X_train, X_rest, y_train, y_rest = train_test_split(X, y, test_size=0.30, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_rest, y_rest, test_size=0.5, random_state=42)

print("Train:", X_train.shape[0], "| Validation:", X_val.shape[0], "| Test:", X_test.shape[0])
```

### Captura (opcional)

<!-- INSERTAR CAPTURA: Tamaños Train / Validation / Test -->
![División datos](capturas/06_division_datos.png)

---

## 5. Modelo 1: Naive Bayes

### Para qué sirve

Naive Bayes en scikit-learn es clasificador; el objetivo es **regresión** (predecir número de casos). Por eso se **discretiza** `total_cases` en intervalos (bins), se entrena **GaussianNB** y luego se convierte la clase predicha al **valor central del intervalo** para obtener una predicción numérica. Se usa **GridSearch** y **RandomizedSearch** con **Cross Validation** y el criterio de calidad es el **MAE en validación**.

### Código explicado (discretización y función de predicción)

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error

n_bins = 15
# Convertir total_cases en etiquetas 0, 1, ..., n_bins-1
y_train_binned = pd.cut(y_train, bins=n_bins, labels=False)
y_val_binned = pd.cut(y_val, bins=n_bins, labels=False)

# Bordes de los intervalos para pasar de clase predicha a valor numérico (centro del intervalo)
_, bin_edges = pd.cut(y_train, bins=n_bins, retbins=True)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

def nb_pred_to_regression(clf, X, bin_centers):
    """Predicción de clase -> valor central del bin (regresión)."""
    pred_class = clf.predict(X)
    pred_class = np.clip(pred_class, 0, len(bin_centers)-1)
    return bin_centers[pred_class.astype(int)]
```

### Código explicado (GridSearch y RandomSearch)

```python
# GridSearch: barrido sobre var_smoothing (suavizado de varianza en GaussianNB)
param_grid_nb = {'var_smoothing': np.logspace(-12, -8, 20)}
grid_nb = GridSearchCV(GaussianNB(), param_grid_nb, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
grid_nb.fit(X_train, y_train_binned)

# Criterio: mejor MAE en validación (se compara con el modelo de RandomSearch y se elige el mejor)
# ... (RandomSearch con loguniform) ...
# best_nb = elegido entre grid_nb y random_nb según MAE en validación
```

---

## 6. Modelo 2: KNN

### Para qué sirve

**KNeighborsRegressor** predice el valor objetivo promediando los valores de los **k** vecinos más cercanos. Se prueban distintos **k**, pesos (**uniform** vs **distance**) y norma (**p=1** Manhattan, **p=2** Euclidea) con **GridSearch** y **RandomizedSearch** y **Cross Validation**; el criterio de calidad es el **MAE en validación**.

### Código explicado

```python
from sklearn.neighbors import KNeighborsRegressor

param_grid_knn = {
    'n_neighbors': list(range(3, 51, 2)),  # k impar desde 3 hasta 49
    'weights': ['uniform', 'distance'],     # todos iguales o ponderar por distancia
    'p': [1, 2]                             # Manhattan (1) o Euclidea (2)
}
grid_knn = GridSearchCV(KNeighborsRegressor(), param_grid_knn, cv=5,
                        scoring='neg_mean_absolute_error', n_jobs=-1)
grid_knn.fit(X_train, y_train)

# Se compara con RandomizedSearch y se elige best_knn según MAE en validación
```

---

## 7. Modelo 3: Random Forest

### Para qué sirve

**RandomForestRegressor** combina muchos árboles de decisión y promedia sus predicciones. Se ajustan **n_estimators**, **max_depth**, **min_samples_split** y **min_samples_leaf** con **GridSearch** y **RandomizedSearch** y **Cross Validation**; el criterio de calidad es el **MAE en validación**.

### Código explicado

```python
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

param_grid_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_rf = GridSearchCV(RandomForestRegressor(random_state=42), param_grid_rf, cv=5,
                       scoring='neg_mean_absolute_error', n_jobs=-1)
grid_rf.fit(X_train, y_train)

# RandomSearch con randint para explorar más combinaciones; best_rf = el de mejor MAE en validación
```

---

## 8. Comparativa de modelos (gráficos)

### Para qué sirve

Integrar **gráficos** para comparar el rendimiento de los tres modelos: barras de MAE en validación y boxplot de MAE en Cross Validation.

### Código explicado (barras de MAE)

```python
mae_nb_final = mean_absolute_error(y_val, nb_pred_to_regression(best_nb, X_val, bin_centers))
mae_knn_final = mean_absolute_error(y_val, best_knn.predict(X_val))
mae_rf_final = mean_absolute_error(y_val, best_rf.predict(X_val))
modelos = ['Naive Bayes', 'KNN', 'Random Forest']
maes = [mae_nb_final, mae_knn_final, mae_rf_final]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(modelos, maes, color=['#2ecc71', '#3498db', '#9b59b6'])
ax.set_ylabel('MAE (validación)')
ax.set_title('Comparativa de modelos: MAE en conjunto de validación')
for b, v in zip(bars, maes):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.2, f'{v:.2f}', ha='center', fontsize=11)
plt.tight_layout()
plt.show()
```

### Hueco para captura – Comparativa MAE validación

<!-- INSERTAR CAPTURA: Gráfico de barras MAE por modelo (validación) -->
![Comparativa MAE validación](capturas/08_comparativa_mae_validacion.png)

---

### Código explicado (boxplot Cross Validation)

```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)
# Para Naive Bayes: MAE en regresión (convirtiendo clase -> centro del bin) en cada fold
cv_nb_mae = []
for tr, te in kf.split(X_train):
    pred = nb_pred_to_regression(best_nb, X_train.iloc[te], bin_centers)
    cv_nb_mae.append(mean_absolute_error(y_train.iloc[te], pred))
cv_knn = -cross_val_score(best_knn, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
cv_rf = -cross_val_score(best_rf, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')

fig, ax = plt.subplots(figsize=(9, 5))
ax.boxplot([cv_nb_mae, cv_knn, cv_rf], labels=modelos, patch_artist=True)
ax.set_ylabel('MAE')
ax.set_title('Distribución de MAE en Cross Validation (5 folds)')
plt.tight_layout()
plt.show()
```

### Hueco para captura – Boxplot Cross Validation

<!-- INSERTAR CAPTURA: Boxplot MAE en Cross Validation por modelo -->
![Boxplot CV](capturas/09_boxplot_cv.png)

---

## 9. Predicción y validación

### Para qué sirve

Evaluar los tres modelos en el **conjunto de test** (MAE) y usar **gráficos** para entender la precisión: real vs predicho y distribución de los errores (residuos).

### Código explicado (evaluación en test)

```python
y_test_pred_nb = nb_pred_to_regression(best_nb, X_test, bin_centers)
y_test_pred_knn = best_knn.predict(X_test)
y_test_pred_rf = best_rf.predict(X_test)

mae_test_nb = mean_absolute_error(y_test, y_test_pred_nb)
mae_test_knn = mean_absolute_error(y_test, y_test_pred_knn)
mae_test_rf = mean_absolute_error(y_test, y_test_pred_rf)

# Modelo para la competición: el de menor MAE en test
best_model_name = min([
    ('Naive Bayes', mae_test_nb),
    ('KNN', mae_test_knn),
    ('Random Forest', mae_test_rf)
], key=lambda x: x[1])
print("Modelo seleccionado para la competición:", best_model_name[0])
```

### Código explicado (real vs predicho)

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, name, y_pred in zip(axes, modelos, [y_test_pred_nb, y_test_pred_knn, y_test_pred_rf]):
    ax.scatter(y_test, y_pred, alpha=0.5, s=20)
    ax.plot([0, y_test.max()], [0, y_test.max()], 'r--', label='Ideal')  # línea perfecta
    ax.set_xlabel('Real')
    ax.set_ylabel('Predicho')
    ax.set_title(name)
    ax.legend()
plt.suptitle('Precisión de los resultados: Real vs Predicho (conjunto test)')
plt.tight_layout()
plt.show()
```

### Hueco para captura – Real vs predicho (test)

<!-- INSERTAR CAPTURA: Tres scatter Real vs Predicho (Naive Bayes, KNN, Random Forest) -->
![Real vs Predicho](capturas/10_real_vs_predicho.png)

---

### Código explicado (distribución de errores)

```python
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, name, y_pred in zip(axes, modelos, [y_test_pred_nb, y_test_pred_knn, y_test_pred_rf]):
    resid = y_test.values - y_pred   # error = real - predicho
    ax.hist(resid, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--')  # línea en error 0
    ax.set_xlabel('Error (real - predicho)')
    ax.set_title(name)
plt.suptitle('Distribución de errores en test')
plt.tight_layout()
plt.show()
```

### Hueco para captura – Distribución de errores

<!-- INSERTAR CAPTURA: Histogramas de residuos por modelo -->
![Distribución errores](capturas/11_distribucion_errores.png)

---

## 10. Generación del fichero de submit

### Para qué sirve

Predecir **total_cases** para las filas del `features_test` de la competición, redondear a enteros no negativos y guardar un CSV con columnas **city, year, weekofyear, total_cases** para subirlo a DrivenData.

### Código explicado

```python
X_test_comp = X_test_norm[selected_features]

if best_model_name[0] == 'Naive Bayes':
    pred_submit = nb_pred_to_regression(best_nb, X_test_comp, bin_centers)
elif best_model_name[0] == 'KNN':
    pred_submit = best_knn.predict(X_test_comp)
else:
    pred_submit = best_rf.predict(X_test_comp)

# La competición pide números enteros no negativos (casos)
pred_submit = np.round(np.maximum(0, pred_submit)).astype(int)

# Formato exigido por DrivenData
submit = features_test[['city', 'year', 'weekofyear']].copy()
submit['total_cases'] = pred_submit

out_path = 'submission.csv'
submit.to_csv(out_path, index=False)
print("Fichero guardado:", out_path)
print(submit.head(10))
```

### Captura (opcional)

<!-- INSERTAR CAPTURA: Salida de submit (head del CSV o mensaje "Fichero guardado") -->
![Submit](capturas/12_submission.png)

---

## Cómo usar los huecos de capturas

1. Crea una carpeta **`capturas`** dentro de `MOSQUITO` (o la ruta que uses).
2. Al ejecutar el notebook, guarda cada gráfico (botón “Save” en la ventana de la figura o `plt.savefig('capturas/03_correlacion_target.png')` antes de `plt.show()`).
3. En este Markdown, las imágenes están referenciadas como `capturas/XX_nombre.png`. Si usas otros nombres, cambia la ruta en cada `![](capturas/...)`.
4. Si en lugar de imágenes usas un editor que no muestra imágenes, deja los comentarios `<!-- INSERTAR CAPTURA: ... -->` como recordatorio de qué captura va en cada sitio.

### Guardar las gráficas desde el notebook

Puedes añadir antes de cada `plt.show()` una línea para guardar la figura en `capturas/`:

```python
# Ejemplo: antes de plt.show() en el gráfico de correlación
plt.savefig('capturas/03_correlacion_target.png', dpi=150, bbox_inches='tight')
plt.show()
```

Nombres sugeridos para cada gráfico (según el documento):

| Gráfico | Nombre sugerido |
|---------|------------------|
| Salida importación | `01_importacion.png` |
| Preparación / describe | `02_preparacion.png` |
| Correlación con total_cases | `03_correlacion_target.png` |
| Importancia (árbol) | `04_importancia_arbol.png` |
| Matriz de correlación | `05_matriz_correlacion.png` |
| Tamaños train/val/test | `06_division_datos.png` |
| Comparativa MAE validación | `08_comparativa_mae_validacion.png` |
| Boxplot Cross Validation | `09_boxplot_cv.png` |
| Real vs predicho | `10_real_vs_predicho.png` |
| Distribución de errores | `11_distribucion_errores.png` |
| Submit / head CSV | `12_submission.png` |
