# %%
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# %%
# Cargar el dataset 
data = pd.read_excel('dataset.xlsx',sheet_name='DNM')

# %%
# Separa las características (columnas 0 a 24) de la etiqueta (última columna)
X = data.iloc[:, :-1]  # Características
y = data.iloc[:, -1]   # Etiqueta

# %%
# Normaliza las características en el rango de 0 a 1
scaler = MinMaxScaler(feature_range=(0.1, 1))
X_normalized = scaler.fit_transform(X)

# %%
# Crea un nuevo DataFrame con las características normalizadas y la etiqueta
normalized_data = pd.DataFrame(X_normalized, columns=X.columns)
normalized_data['Etiqueta'] = y  # Agrega la columna de etiqueta al DataFrame

# %%
# Guarda el DataFrame normalizado en un nuevo archivo excel
normalized_data.to_excel('datos_normalizados.xlsx', index=False)


