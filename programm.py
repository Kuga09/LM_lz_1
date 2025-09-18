# Импортируем необходимые библиотеки
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import pandas as pd


# Загружаем данные из таблицы
data = pd.read_csv('table.csv')

# Извлекаем данные для обучени, объединяем категорию трат и стоимость
expenses_data = data['spending_category'] + " " + data['price'].astype(str) 
labels = data['importance']

# Разделяем данные
text_train, text_test, y_train, y_test = train_test_split(
    expenses_data, labels, test_size=0.2, random_state=42
)

# Создаем векторизатор и модель наивного Байеса 
pipe = make_pipeline(
    CountVectorizer(ngram_range=(1,2), min_df=1),
    MultinomialNB()
)

# Обучаем модель
pipe.fit(text_train, y_train)

# Делаем предсказания на тесте
y_pred = pipe.predict(text_test)

# Оцениваем качество предсказания
print(f"Accuracy: {accuracy_score(y_test, y_pred):.1f}")

# Проверяем на новых примерах
new_data = [
    "Хлеб молоко продукты семья 2000 рублей",
    "Сигареты табак курение 2000 рублей",
    "Лекарства аспирин температура 800 рублей",
    "Суши роллы ресторан друзья 3500 рублей",
    "Коммунальные платежи свет газ 7000 рублей",
    "Компьютер игры развлечения 15000 рублей",
    "Детская одежда школа форма 5000 рублей"
]

for example in new_data:
    prediction = pipe.predict([example])[0]
    if prediction == 1:
        result = "важно"
    else:
        result = "неважно"
    print(f"{result} - {example}")
