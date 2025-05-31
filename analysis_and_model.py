import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)

def analysis_and_model_page():
    st.title("Анализ данных и обучение модели")
    st.markdown(
        """
        **Задача:** бинарная классификация: предсказание отказа оборудования (Target = 1) или нет (Target = 0).
        """
    )

    st.header("1. Загрузка и предобработка данных")
    st.markdown("Можно загрузить CSV-файл (AI4I 2020) вручную или использовать встроенный загрузчик.")
    uploaded_file = st.file_uploader("Загрузите CSV (необязательно)", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
    else:
        st.info("Загружаем датасет через ucimlrepo...")
        from ucimlrepo import fetch_ucirepo
        ds = fetch_ucirepo(id=601)
        data = pd.concat([ds.data.features, ds.data.targets], axis=1)

    drop_cols = ["UDI", "Product ID", "TWF", "HDF", "PWF", "OSF", "RNF"]
    data = data.drop(columns=[c for c in drop_cols if c in data.columns])

    if "Type" in data.columns:
        label_enc_type = LabelEncoder()
        data["Type"] = label_enc_type.fit_transform(data["Type"])
    else:
        label_enc_type = LabelEncoder().fit(["L", "M", "H"])

    if data.isnull().sum().sum() > 0:
        st.warning("В данных есть пропущенные значения. Сейчас мы их удалим.")
        data = data.dropna()

    st.write("Пример данных после предобработки:", data.head())

    st.subheader("Масштабирование числовых признаков")

    numeric_cols = data.select_dtypes(include=["int64", "float64"]).columns.tolist()
    if "Machine failure" in numeric_cols:
        numeric_cols.remove("Machine failure")

    if numeric_cols:
        scaler = StandardScaler()
        data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        st.write(f"Масштабированные столбцы: {numeric_cols}")
        st.write("Данные после масштабирования (первые 5 строк):")
        st.write(data.head())
    else:
        st.warning("Не найдено числовых признаков для масштабирования — пропускаем этот шаг.")
        scaler = None
        numeric_cols = []

    st.subheader("Разделение на обучающую и тестовую выборки (80/20)")
    if "Machine failure" not in data.columns:
        st.error("Столбец 'Machine failure' не найден в данных. Проверьте структуру DataFrame.")
        return

    X = data.drop(columns=["Machine failure"])
    y = data["Machine failure"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write(f"Размер обучающей выборки: {X_train.shape[0]} строк, тестовой: {X_test.shape[0]} строк")

    st.header("Обучение моделей")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42
        ),
    }
    if st.checkbox("Обучить SVM (медленнее)"):
        from sklearn.svm import SVC
        models["SVM (Linear)"] = SVC(kernel="linear", probability=True, random_state=42)

    trained_models = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        trained_models[name] = mdl
        st.success(f"Модель `{name}` обучена.")

    st.header("Оценка моделей")
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Случайная модель")

    for name, mdl in trained_models.items():
        y_pred = mdl.predict(X_test)
        if hasattr(mdl, "predict_proba"):
            y_proba = mdl.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_proba)
        else:
            y_proba = mdl.decision_function(X_test)
            roc_auc = roc_auc_score(y_test, y_proba)

        acc = accuracy_score(y_test, y_pred)
        st.subheader(f"Модель: {name}")
        st.write(f"- Accuracy: **{acc:.3f}**")
        st.write("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Предсказано")
        ax.set_ylabel("Истинно")
        st.pyplot(fig)
        st.text(classification_report(y_test, y_pred))

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.2f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-кривые моделей")
    plt.legend()
    st.pyplot(plt.gcf())

    st.header("Предсказание для новых данных")
    with st.form("prediction_form"):
        st.write("Введите входные параметры:")
        product_type = st.selectbox("Type (L/M/H)", ["L", "M", "H"])
        air_temp = st.number_input("Air temperature [K]", value=300.0)
        proc_temp = st.number_input("Process temperature [K]", value=310.0)
        rot_speed = st.number_input("Rotational speed [rpm]", value=1000)
        torque = st.number_input("Torque [Nm]", value=40.0)
        tool_wear = st.number_input("Tool wear [min]", value=100)

        submit = st.form_submit_button("Предсказать")

        if submit:
            if not numeric_cols or scaler is None:
                st.error("Невозможно выполнить предсказание: отсутствуют числовые признаки или не инициализирован scaler.")
            else:
                try:
                    type_num = int(label_enc_type.transform([product_type])[0])
                except:
                    type_num = int(LabelEncoder().fit(["L", "M", "H"]).transform([product_type])[0])

                row = {}
                if "Type" in X.columns:
                    row["Type"] = type_num

                for col in numeric_cols:
                    if col == "Type":
                        continue
                    elif col == "Air temperature [K]":
                        row[col] = air_temp
                    elif col == "Process temperature [K]":
                        row[col] = proc_temp
                    elif col == "Rotational speed [rpm]":
                        row[col] = rot_speed
                    elif col == "Torque [Nm]":
                        row[col] = torque
                    elif col == "Tool wear [min]":
                        row[col] = tool_wear
                    else:
                        row[col] = 0

                input_df = pd.DataFrame([row])
                input_df = input_df.reindex(columns=X.columns, fill_value=0)
                input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

                if "Random Forest" in trained_models:
                    best_model = trained_models["Random Forest"]
                else:
                    best_model = list(trained_models.values())[0]

                pred = best_model.predict(input_df)[0]
                if hasattr(best_model, "predict_proba"):
                    proba = best_model.predict_proba(input_df)[0][1]
                else:
                    proba = None

                st.write(f"**Предсказание отказа (0 — нет, 1 — да): {int(pred)}**")
                if proba is not None:
                    st.write(f"Вероятность отказа: {proba:.2f}")
                else:
                    st.warning("Модель не поддерживает вывод вероятности.")
