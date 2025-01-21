import streamlit as st
import pandas as pd
from catboost import CatBoostRegressor
from sentence_transformers import SentenceTransformer


def create_and_apply_mapping(df, columns):
    mappings = {}
    for column in columns:
        unique_values = df[column].unique()
        mapping = {value: idx for idx, value in enumerate(unique_values)}
        mappings[column] = mapping
        df[column] = df[column].map(mapping)
    return mappings


def apply_mapping(df, mappings):
    for column, mapping in mappings.items():
        df[column] = df[column].map(mapping).fillna(-1).astype(int)
    return df


def combine_text(row):
    text_fields = [str(row[col]).lower() for col in ["name", "description", "employer_name", "key_skills"]]
    return " ".join(text_fields)


st.title("Инференс модели для предсказания зарплаты")

mappings_global = {
    "schedule": {"fullDay": 0, "partTime": 1, "shift": 2, "flexible": 3, "remote": 4},
    "experience": {"noExperience": 0, "between1And3": 1, "between3And6": 2, "moreThan6": 3},
    "employment": {"full": 0, "part": 1, "project": 2, "volunteer": 3, "internship": 4},
    "region_name": {
        "Томская область": 0,
        "Москва": 1,
        "Московская область": 2,
        "Санкт-Петербург": 3,
        "Тюменская область": 4,
        "Новосибирская область": 5,
        "Пензенская область": 6,
        "Республика Татарстан": 7,
        "Республика Башкортостан": 8,
        "Калининградская область": 9,
        "Ленинградская область": 10,
        "Псковская область": 11,
    },
}


def manual_input_and_predict():
    st.write("Введите данные для прогноза:")
    name = st.text_input("Название вакансии:")
    description = st.text_area("Описание вакансии:")
    employer_name = st.text_input("Название работодателя:")
    schedule = st.selectbox("График работы:", list(mappings_global["schedule"].keys()))
    key_skills = st.text_input("Ключевые навыки (через запятую):")
    experience = st.selectbox("Опыт работы:", list(mappings_global["experience"].keys()))
    employment = st.selectbox("Тип занятости:", list(mappings_global["employment"].keys()))
    region_name = st.selectbox("Регион:", list(mappings_global["region_name"].keys()))

    if any([name, description, employer_name, schedule, key_skills, experience, employment, region_name]):
        data = {
            "name": name,
            "description": description,
            "employer_name": employer_name,
            "schedule": schedule,
            "key_skills": key_skills,
            "experience": experience,
            "employment": employment,
            "region_name": region_name,
        }
        df = pd.DataFrame([data])

        df["schedule"] = df["schedule"].map(mappings_global["schedule"])
        df["experience"] = df["experience"].map(mappings_global["experience"])
        df["employment"] = df["employment"].map(mappings_global["employment"])
        df["region_name"] = df["region_name"].map(mappings_global["region_name"])

        df["combined_text"] = df.apply(combine_text, axis=1)

        text_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = text_model.encode(df["combined_text"].tolist())
        embeddings_df = pd.DataFrame(embeddings, columns=[f"embedding_{i}" for i in range(embeddings.shape[1])])

        df = pd.concat([df, embeddings_df], axis=1).drop(
            columns=["combined_text", "name", "description", "employer_name", "key_skills"]
        )

        model = CatBoostRegressor()
        model.load_model("catboost_model.cbm")

        prediction = model.predict(df)

        st.write(f"Предсказанная зарплата: {prediction[0]:.2f}")


manual_input_and_predict()

uploaded_file = st.file_uploader("Загрузите CSV-файл с данными", type=["csv"])
data = pd.DataFrame()
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Загруженные данные:", data.head())

if not data.empty:
    st.write("Предобработка данных...")
    encoded_columns = ["schedule", "experience", "employment", "region_name"]
    mappings = create_and_apply_mapping(data, encoded_columns)

    data["combined_text"] = data.apply(combine_text, axis=1)

    st.write("Создание эмбеддингов...")
    text_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = text_model.encode(data["combined_text"].tolist())
    embeddings_df = pd.DataFrame(embeddings, columns=[f"embedding_{i}" for i in range(embeddings.shape[1])])

    data = pd.concat([data, embeddings_df], axis=1).drop(
        columns=["combined_text", "name", "description", "employer_name", "key_skills"]
    )

    st.write("Загрузка модели...")
    model = CatBoostRegressor()
    model.load_model("catboost_model.cbm")

    st.write("Выполнение предсказаний...")
    features = data.drop(columns=["salary"])
    predictions = model.predict(features)

    data["predicted_salary"] = predictions
    st.write("Результаты предсказания:", data[["salary", "predicted_salary"]])

    st.download_button(
        label="Скачать результаты",
        data=data.to_csv(index=False).encode("utf-8"),
        file_name="predictions.csv",
        mime="text/csv",
    )
