import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Загрузка данных
data = pd.read_csv('data_merged.csv')

# Определение признаков и целевой переменной
X = data[['Bm1_N', 'Bm1_S', 'Fm1_B', 'Bm2_N', 'Bm2_S', 'Fm2_B']]
y = data[['Bpair_N', 'Bpair_S', 'Fpair_N', 'Fpair_S']]

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели (например, Random Forest)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Оценка модели
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))

# Пример предсказания для новых данных
new_data = pd.DataFrame([[234, 233, 5.94, 233, 236, 5.96]], 
                        columns=['Bm1_N', 'Bm1_S', 'Fm1_B', 'Bm2_N', 'Bm2_S', 'Fm2_B'])
predicted_values = model.predict(new_data)
print("Предсказанные значения:", predicted_values)
