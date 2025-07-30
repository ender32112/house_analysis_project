# house_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import os

# --- 1. НАСТРОЙКА СТРАНИЦЫ ---
st.set_page_config(
    page_title="Анализ недвижимости Сиэттла",
    page_icon="🏠",
    layout="wide"
)

# --- 2. ЗАГОЛОВОК ---
st.title("🏠 Анализ рынка недвижимости Сиэттла")
st.markdown("---")


# --- 3. ЗАГРУЗКА ДАННЫХ ---
@st.cache_data
def load_data():
    # Попробуем разные пути к файлу
    possible_paths = [
        'data/kc_house_data.csv',
        '../data/kc_house_data.csv',
        'kc_house_data.csv'
    ]

    for path in possible_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                return df
            except Exception as e:
                st.warning(f"Ошибка при загрузке {path}: {e}")
                continue

    # Если файл не найден ни по одному пути
    st.error("Файл данных 'kc_house_data.csv' не найден. Проверьте структуру проекта.")
    st.stop()
    return None


# Загрузка данных
df = load_data()

if df is not None and not df.empty:
    # --- 4. БОКОВАЯ ПАНЕЛЬ (ФИЛЬТРЫ) ---
    st.sidebar.header("📊 Фильтры")

    # Фильтр по цене
    price_range = st.sidebar.slider(
        "Диапазон цен ($)",
        int(df['price'].min()),
        int(df['price'].max()),
        (int(df['price'].min()), int(df['price'].max())),
        step=10000
    )

    # Фильтр по количеству спален
    bedrooms_options = sorted([x for x in df['bedrooms'].unique() if pd.notna(x)])
    bedrooms = st.sidebar.multiselect(
        "Количество спален",
        options=bedrooms_options,
        default=bedrooms_options
    )

    # Фильтрация данных
    if bedrooms:  # Проверяем, что выбраны спальни
        filtered_df = df[
            (df['price'] >= price_range[0]) &
            (df['price'] <= price_range[1]) &
            (df['bedrooms'].isin(bedrooms))
            ]
    else:
        # Если ничего не выбрано, показываем все данные в диапазоне цен
        filtered_df = df[
            (df['price'] >= price_range[0]) &
            (df['price'] <= price_range[1])
            ]
else:
    st.error("Не удалось загрузить данные")
    st.stop()

# --- 5. МЕТРИКИ НА ГЛАВНОЙ СТРАНИЦЕ ---
if not filtered_df.empty:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего домов", f"{len(filtered_df):,}")
    col2.metric("Средняя цена", f"${filtered_df['price'].mean():,.0f}")
    col3.metric("Средняя площадь", f"{filtered_df['sqft_living'].mean():.0f} sqft")
    col4.metric("Средний рейтинг", f"{filtered_df['grade'].mean():.1f}")
else:
    st.warning("Нет данных, соответствующих фильтрам")
    st.stop()

st.markdown("---")

# --- 6. ВКЛАДКИ ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Обзор", "📊 Анализ", "🤖 ML", "🔮 Прогноз"])

# --- ВКЛАДКА 1: Обзор ---
with tab1:
    st.header("Обзор рынка недвижимости")

    # Распределение цен
    fig_price_dist = px.histogram(
        filtered_df,
        x='price',
        nbins=50,
        title='Распределение цен на дома',
        labels={'price': 'Цена ($)', 'count': 'Количество'},
        color_discrete_sequence=['skyblue']
    )
    fig_price_dist.update_layout(showlegend=False)
    st.plotly_chart(fig_price_dist, use_container_width=True)

    # Корреляции
    corr_features = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'grade']
    corr_matrix = filtered_df[corr_features].corr()

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='RdBu_r',
        title="Корреляционная матрица"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# --- ВКЛАДКА 2: Анализ ---
with tab2:
    st.header("Анализ факторов ценообразования")

    col1, col2 = st.columns(2)

    with col1:
        # Цена vs Жилая площадь
        fig_scatter = px.scatter(
            filtered_df,
            x='sqft_living',
            y='price',
            color='grade',
            size='bedrooms',
            title='Цена vs Жилая площадь',
            labels={'sqft_living': 'Жилая площадь (sqft)', 'price': 'Цена ($)', 'grade': 'Оценка',
                    'bedrooms': 'Спален'},
            hover_data=['yr_built']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    with col2:
        # Распределение по оценкам
        grade_counts = filtered_df['grade'].value_counts().sort_index()
        fig_bar = px.bar(
            x=grade_counts.index,
            y=grade_counts.values,
            title='Распределение домов по оценке качества',
            labels={'x': 'Оценка качества', 'y': 'Количество домов'},
            color=grade_counts.index,
            color_continuous_scale='Viridis'
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    # Цена по годам постройки
    yearly_avg = filtered_df.groupby('yr_built')['price'].mean().reset_index()
    fig_line = px.line(
        yearly_avg,
        x='yr_built',
        y='price',
        title='Средняя цена по годам постройки',
        labels={'yr_built': 'Год постройки', 'price': 'Средняя цена ($)'},
        markers=True
    )
    st.plotly_chart(fig_line, use_container_width=True)

# --- ВКЛАДКА 3: ML ---
with tab3:
    st.header("Машинное обучение")

    # Подготовка данных
    features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'waterfront', 'view', 'condition', 'grade', 'yr_built']
    X = filtered_df[features]
    y = filtered_df['price']

    # Удаление строк с пропущенными значениями
    data_for_ml = pd.concat([X, y], axis=1).dropna()
    X_clean = data_for_ml[features]
    y_clean = data_for_ml['price']

    if len(X_clean) < 10:  # Минимальный размер для разбиения
        st.warning("Недостаточно данных для обучения модели. Попробуйте изменить фильтры.")
    else:
        X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)

        # Масштабирование для линейной регрессии
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Модели
        col1, col2 = st.columns(2)

        # --- Линейная регрессия ---
        with col1:
            st.subheader("Линейная регрессия")

            with st.spinner('Обучение модели...'):
                lr_model = LinearRegression()
                lr_model.fit(X_train_scaled, y_train)
                lr_pred = lr_model.predict(X_test_scaled)

            lr_mae = mean_absolute_error(y_test, lr_pred)
            lr_r2 = r2_score(y_test, lr_pred)

            st.metric("R²", f"{lr_r2:.3f}")
            st.metric("MAE", f"${lr_mae:,.0f}")

            # Важность признаков (коэффициенты)
            feature_importance_lr = pd.DataFrame({
                'feature': features,
                'importance': np.abs(lr_model.coef_)
            }).sort_values('importance', ascending=True)

            fig_lr_importance = px.bar(
                feature_importance_lr.tail(10),
                x='importance',
                y='feature',
                title='Важность признаков (Линейная регрессия)',
                orientation='h'
            )
            st.plotly_chart(fig_lr_importance, use_container_width=True)

        # --- Random Forest ---
        with col2:
            st.subheader("Random Forest")

            with st.spinner('Обучение модели...'):
                rf_model = RandomForestRegressor(n_estimators=50, random_state=42,
                                                 max_depth=10)  # Ограничим глубину для скорости
                rf_model.fit(X_train, y_train)  # RF не требует масштабирования
                rf_pred = rf_model.predict(X_test)

            rf_mae = mean_absolute_error(y_test, rf_pred)
            rf_r2 = r2_score(y_test, rf_pred)

            st.metric("R²", f"{rf_r2:.3f}")
            st.metric("MAE", f"${rf_mae:,.0f}")

            # Важность признаков
            feature_importance_rf = pd.DataFrame({
                'feature': features,
                'importance': rf_model.feature_importances_
            }).sort_values('importance', ascending=True)

            fig_rf_importance = px.bar(
                feature_importance_rf.tail(10),
                x='importance',
                y='feature',
                title='Важность признаков (Random Forest)',
                orientation='h'
            )
            st.plotly_chart(fig_rf_importance, use_container_width=True)

        # --- Сравнение моделей ---
        st.subheader("Сравнение моделей")
        comparison_data = pd.DataFrame({
            'Модель': ['Линейная регрессия', 'Random Forest'],
            'R²': [lr_r2, rf_r2],
            'MAE': [lr_mae, rf_mae]
        })
        st.table(comparison_data.set_index('Модель'))

# --- ВКЛАДКА 4: Прогноз ---
with tab4:
    st.header("Прогнозирование цены дома")

    st.subheader("Введите параметры дома:")

    col1, col2, col3 = st.columns(3)

    with col1:
        bedrooms_input = st.number_input("Количество спален", min_value=1, max_value=20, value=3, step=1)
        bathrooms_input = st.number_input("Количество ванных", min_value=1.0, max_value=10.0, value=2.0, step=0.25)
        sqft_living_input = st.number_input("Жилая площадь (sqft)", min_value=100, max_value=20000, value=2000,
                                            step=100)
        sqft_lot_input = st.number_input("Общая площадь участка (sqft)", min_value=500, max_value=1000000, value=5000,
                                         step=500)

    with col2:
        floors_input = st.number_input("Этажность", min_value=1.0, max_value=5.0, value=1.0, step=0.5)
        waterfront_input = st.selectbox("Выход к воде", options=["Нет", "Да"])
        view_input = st.slider("Вид (0-4)", 0, 4, 0)
        condition_input = st.slider("Состояние (1-5)", 1, 5, 3)

    with col3:
        grade_input = st.slider("Оценка качества (1-13)", 1, 13, 7)
        yr_built_input = st.number_input("Год постройки", min_value=1900, max_value=2023, value=2000, step=1)

    # Прогноз
    if st.button("🔮 Рассчитать прогнозную цену", type="primary"):
        # Подготовка данных для прогноза
        waterfront_val = 1 if waterfront_input == "Да" else 0

        # Подготовка данных
        input_data = pd.DataFrame({
            'bedrooms': [bedrooms_input],
            'bathrooms': [bathrooms_input],
            'sqft_living': [sqft_living_input],
            'sqft_lot': [sqft_lot_input],
            'floors': [floors_input],
            'waterfront': [waterfront_val],
            'view': [view_input],
            'condition': [condition_input],
            'grade': [grade_input],
            'yr_built': [yr_built_input]
        })

        # Проверяем, существуют ли модели (если вкладка ML была открыта)
        try:
            # Масштабирование для линейной регрессии
            input_data_scaled = scaler.transform(input_data)

            # Прогноз с обеих моделей
            lr_prediction = lr_model.predict(input_data_scaled)[0]
            rf_prediction = rf_model.predict(input_data)[0]

            # Отображение результатов
            st.success("✅ Прогноз рассчитан!")
            col1, col2, col3 = st.columns(3)
            col1.metric("Линейная регрессия", f"${lr_prediction:,.0f}")
            col2.metric("Random Forest", f"${rf_prediction:,.0f}")
            col3.metric("Средняя оценка", f"${(lr_prediction + rf_prediction) / 2:,.0f}")

            # Визуализация прогноза
            pred_df = pd.DataFrame({
                'Модель': ['Линейная регрессия', 'Random Forest', 'Среднее'],
                'Прогноз': [lr_prediction, rf_prediction, (lr_prediction + rf_prediction) / 2]
            })
            fig_pred = px.bar(pred_df, x='Модель', y='Прогноз', title="Сравнение прогнозов моделей", color='Модель')
            st.plotly_chart(fig_pred, use_container_width=True)

        except NameError:
            st.error("Модели еще не обучены. Перейдите на вкладку '🤖 ML' и нажмите кнопку для обучения моделей.")

# --- ИНФОРМАЦИЯ В БОКОВОЙ ПАНЕЛИ ---
st.sidebar.markdown("---")
st.sidebar.info(f"""
📊 **Информация о данных:**
- Всего записей: {len(df):,}
- Период: 2014-2015 гг.
- Регион: Сиэттл, США
""")

st.sidebar.markdown("---")
st.sidebar.markdown("🏠 Анализ рынка недвижимости")