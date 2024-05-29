# %%
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# %%
# Cargar el dataset 
datos = pd.read_excel('datos_normalizados.xlsx')

# %%
# Extraer las características (features) y las etiquetas
# Asume que las características están en columnas específicas y las etiquetas en la última columna
X = datos.iloc[:, :-1]  # Todas las columnas excepto la última son características
y = datos.iloc[:, -1]   # Última columna son las etiquetas

# %%
# Inicializar y ajustar el modelo t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)

# %%
# Crear un DataFrame con los datos reducidos a 2 dimensiones y las etiquetas
df_embedded = pd.DataFrame(data=X_embedded, columns=['Dimensión_1', 'Dimensión_2'])
df_embedded['Etiqueta'] = y

# %%
# Graficar los datos coloreando por etiqueta
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Dimensión_1', y='Dimensión_2', hue='Etiqueta', data=df_embedded, palette='viridis')
plt.title('Visualización t-SNE con Etiquetas')
plt.xlabel('Dimensión 1')
plt.ylabel('Dimensión 2')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.show()


