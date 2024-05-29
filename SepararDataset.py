# %%
import pandas as pd
from sklearn.model_selection import train_test_split


# %%
# Cargar el dataset 
data = pd.read_excel('datos_normalizados.xlsx')

# %%
# Obtener el número de columnas menos 1 para la última columna de etiquetas
num_columnas = len(data.columns)
columna_etiquetas = data.columns[num_columnas - 1]  # Última columna

# %%
# Filtrar las etiquetas deseadas (por ejemplo, 1 y 2)
etiquetas_deseadas = [1, 2]
subset_1 = data[data[columna_etiquetas].isin(etiquetas_deseadas)]
print(subset_1)

# %%
# Convertir etiquetas 1 y 2 a 0 y 1
subset_1[columna_etiquetas] = subset_1[columna_etiquetas].replace({1: 0, 2: 1})

# Mostrar el subset con las etiquetas modificadas
print(subset_1)


# %%
# Guardar subset_1 en un archivo Excel
nombre_archivo_subset_1 = 'dataset_separado.xlsx'  # Nombre del archivo para subset_1
subset_1.to_excel(nombre_archivo_subset_1, index=False)  # Guardar subset_1 sin el índice



