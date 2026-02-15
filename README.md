# Actividad 3.6 - DengAI: Predicción de la propagación de enfermedades

Competición DrivenData: [DengAI - Predicting Disease Spread](https://www.drivendata.org/competitions/44/dengai-predicting-disease-spread/).

## Contenido del proyecto

- **`data/`** – Datasets: `dengue_labels_train.csv`, `dengue_features_train.csv`, `dengue_features_test.csv`
- **`DengAI_prediccion_dengue.ipynb`** – Notebook con todo el flujo: importación, preparación, selección de características, train/validation/test, 3 modelos (Naive Bayes, KNN, Random Forest), GridSearch/RandomSearch, CV, predicción y generación de `submission.csv`
- **`requirements.txt`** – Dependencias Python
- **`INFORME_ACT3_6_PLANTILLA.md`** – Plantilla del informe PDF (portada, índice, conclusiones, referencias)

## Cómo ejecutar

1. Crear entorno e instalar dependencias:
   ```bash
   pip install -r requirements.txt
   ```
2. Abrir y ejecutar el notebook:
   ```bash
   jupyter notebook DengAI_prediccion_dengue.ipynb
   ```
3. Tras ejecutar todo, se generará **`submission.csv`** en la raíz del proyecto.

## Origen de los datos (criterio de la actividad)

- **Local:** por defecto el notebook carga desde la carpeta `data/`.
- **Google Drive (Colab):** montar Drive y usar `load_data_local('/content/drive/MyDrive/MOSQUITO/data')`.
- **GitHub:** usar `load_data_from_github('https://raw.githubusercontent.com/USUARIO/REPO/main/data')` con la URL base de los CSV en raw.

## Entrega del informe PDF

1. Nombre del archivo: **`SNS_ACT3_6_NombreApellidos.pdf`**
2. Incluir:
   - Portada (título, asignatura, nombre, enlaces a GitHub y Google Colab)
   - Índice
   - Capturas del código y resultados del notebook
   - Captura de la **valoración/posicionamiento** en la competición DrivenData tras subir `submission.csv`
   - Conclusiones
   - Referencias (web), incluyendo referencias externas no recogidas en el material suministrado

Puedes usar la plantilla **`INFORME_ACT3_6_PLANTILLA.md`** para estructurar el PDF.

## Métrica de la competición

La plataforma DrivenData valora las soluciones con **MAE (Mean Absolute Error)**.
