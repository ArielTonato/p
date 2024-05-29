# %%
import pandas as pd

# %%
# Cargar el dataset 
data = pd.read_excel('datos_normalizados.xlsx')

# %%
 #Obtener las características (todas las columnas excepto la última)
caracteristicas = data.iloc[:, :-1]

# %%
# Obtener la columna de etiquetas
etiquetas = data.iloc[:, -1]

# %%
# Aplicar One-Hot Encoding a la columna de etiquetas
etiquetas_encoded = pd.get_dummies(etiquetas, prefix='Label')

# %%
etiquetas_encoded = etiquetas_encoded.astype(int)
# Concatenar las características originales con las columnas codificadas en One-Hot
data_encoded = pd.concat([caracteristicas, etiquetas_encoded], axis=1)

# %%
# Mostrar el DataFrame con One-Hot Encoding
print(data_encoded)

# %%
data_encoded.to_excel('datos_encoded.xlsx', index=False)


