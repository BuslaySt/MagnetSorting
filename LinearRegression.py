import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Загрузка данных
data = pd.read_csv('data_merged.csv')

# 2. Подготовка данных
X = data[['Bm1_N', 'Bm1_S', 'Fm1_B', 'Bm2_N', 'Bm2_S', 'Fm2_B']]
y = data[['Bpair_S', 'Fpair_N', 'Fpair_S']]

# 3. Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Предсказание и оценка
y_pred = model.predict(X_test)

print("Оценка модели:")
print(f"R²: {r2_score(y_test, y_pred):.3f}")
print(f"MAE: {mean_absolute_error(y_test, y_pred):.3f}")

# 6. Пример предсказания для новых данных
new_data = pd.DataFrame([[11, 234, 233, 5.94, 17, 233, 236, 5.96]], 
                        columns=X.columns)
prediction = model.predict(new_data)
print("\nПредсказание для новых данных:")
print(pd.DataFrame(prediction, columns=y.columns))

# 7. Визуализация важности признаков (коэффициенты)
plt.figure(figsize=(10, 6))
features = X.columns
coef = pd.DataFrame(model.coef_, columns=features, index=y.columns)
coef.mean().sort_values().plot(kind='barh', color='skyblue')
plt.title('Среднее влияние признаков на целевые переменные')
plt.xlabel('Значение коэффициента')
plt.ylabel('Признак')
plt.show()