import streamlit as st
import reveal_slides as rs

def presentation_page():
    st.title("Презентация проекта: Predictive Maintenance")
    st.markdown("Используем `streamlit-reveal-slides` для демонстрации этапов проекта.")

    slides_md = """
# Прогнозирование отказов оборудования

---

## 1. Цель проекта
- Предсказать, произойдет ли отказ машины (binary target).
- Используем датасет AI4I 2020 (UCI).

---

## 2. Данные
- 10 000 записей, 14 признаков.
- Признаки: Air temp, Process temp, Rotational speed, Torque, Tool wear, Type.

---

## 3. Предобработка
- Удалили `UDI`, `Product ID`, признаки отдельных типов отказов (`TWF`, `HDF`, ...).
- LabelEncoder для `Type`.
- StandardScaler для числовых признаков.

---

## 4. Модели
- Logistic Regression  
- Random Forest  
- XGBoost  
- (Опционально SVM)

---

## 5. Оценка качества
- Accuracy, Confusion Matrix, Classification Report  
- ROC-AUC + объединённая ROC‐кривая

---

## 6. Streamlit-приложение
- Две страницы:
  1. **Анализ и модель:** загрузка данных, тренировка, визуализация, предсказание.
  2. **Презентация:** слайдами показываем этапы.

---

## 7. Выводы и улучшения
- Сравнение моделей: RF часто выигрывает, но возможно XGBoost даёт лучшие AUC.  
- Дальнейшие шаги: подбор гиперпараметров, инжиниринг признаков, стратифицированное разбиение и т. д.

---

**Спасибо за внимание!**
    """

    # Параметры презентации
    with st.sidebar:
        st.header("Настройки слайдов")
        theme = st.selectbox("Theme", ["black", "white", "league", "beige", "sky", "night", "serif", "simple", "solarized"])
        height = st.number_input("Высота", value=500)
        transition = st.selectbox("Переход", ["slide", "convex", "concave", "zoom", "none"])
        plugins = st.multiselect("Плагины", ["highlight", "katex", "mathjax2", "mathjax3", "notes", "search", "zoom"], [])

    rs.slides(
        slides_md,
        height=height,
        theme=theme,
        config={"transition": transition, "plugins": plugins},
        markdown_props={"data-separator-vertical": "^--$"},
    )
