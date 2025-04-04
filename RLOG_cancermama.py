import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar el conjunto de datos
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['Class'] = data.target

# Separar características y etiquetas
X = df.drop('Class', axis=1)
y = df['Class']

# División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalización de características
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Entrenar el modelo de regresión logística
model_binary = LogisticRegression(max_iter=1000)
model_binary.fit(X_train, y_train)

# Predicciones
y_pred_binary = model_binary.predict(X_test)

# Métricas
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"\n🔎 Precisión del modelo: {accuracy:.2f}")

print("\n📋 Informe de clasificación:")
print(classification_report(y_test, y_pred_binary, target_names=data.target_names))

# Matriz de confusión
conf_matrix_binary = confusion_matrix(y_test, y_pred_binary)
print("📊 Matriz de confusión:")
print(conf_matrix_binary)

# Visualización
sns.heatmap(conf_matrix_binary, annot=True, fmt='d', cmap="Blues",
            xticklabels=data.target_names, yticklabels=data.target_names)
plt.title('Matriz de Confusión (Regresión Logística)')
plt.xlabel('Predicción')
plt.ylabel('Valor real')
plt.show()
