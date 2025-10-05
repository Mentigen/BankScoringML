"""
Credit Scoring Dashboard
Streamlit приложение для анализа кредитного скоринга и прогнозирования дефолта
"""

import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_recall_curve
)
import os

# Конфигурация страницы
st.set_page_config(
    page_title="Credit Scoring Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Константы
RANDOM_STATE = 42

# Кэширование данных
@st.cache_data
def load_data(uploaded_file):
    """Загрузка данных из файла"""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        return df
    return None

@st.cache_data
def get_data_info(df):
    """Получение информации о данных"""
    info_dict = {
        'Количество строк': df.shape[0],
        'Количество признаков': df.shape[1],
        'Пропущенные значения': df.isnull().sum().sum(),
        'Дубликаты': df.duplicated().sum()
    }
    return info_dict

def check_data_quality(df):
    """Проверка качества данных"""
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    duplicate_count = df.duplicated().sum()
    return missing_values, duplicate_count

def preprocess_data(df, missing_values):
    """Обработка пропущенных значений и кодирование категориальных переменных"""
    df_processed = df.copy()

    if len(missing_values) > 0:
        for col in missing_values.index:
            if df[col].dtype in ['int64', 'float64']:
                fill_value = df[col].mean()
                df_processed[col] = df_processed[col].fillna(fill_value)
            else:
                fill_value = df[col].mode()[0]
                df_processed[col] = df_processed[col].fillna(fill_value)

    # Кодирование категориальных переменных
    for col in df_processed.select_dtypes(include=['object']).columns:
        if col == 'RealEstateLoansOrLines':
            mapping = {chr(65+i): i+1 for i in range(26)}
            df_processed[f'{col}_numeric'] = df_processed[col].map(mapping)
        else:
            unique_values = df_processed[col].unique()
            mapping = {val: idx for idx, val in enumerate(sorted(unique_values))}
            df_processed[f'{col}_numeric'] = df_processed[col].map(mapping)

    return df_processed

@st.cache_resource
def train_model(X, y):
    """Обучение модели"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )

    model = LogisticRegression(max_iter=10000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test

def plot_distribution(df, column, title, color='steelblue'):
    """График распределения"""
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(df[column], kde=True, color=color, ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return fig

def plot_correlation_matrix(df):
    """Корреляционная матрица"""
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = numeric_df.corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm',
                fmt='.2f', mask=mask, ax=ax, linewidths=0.5)
    ax.set_title('Корреляционная матрица', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig

def plot_confusion_matrix(cm):
    """Матрица ошибок"""
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title('Матрица ошибок', fontsize=14, fontweight='bold')
    ax.set_xlabel('Предсказанная метка', fontsize=12)
    ax.set_ylabel('Истинная метка', fontsize=12)
    return fig

def plot_roc_curve(y_test, y_pred_prob):
    """ROC-кривая"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC кривая (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC-кривая', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    return fig, roc_auc

def plot_feature_importance(model, feature_names):
    """Важность признаков"""
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.coef_[0]
    })
    coefficients['Abs_Coefficient'] = abs(coefficients['Coefficient'])
    coefficients = coefficients.sort_values('Abs_Coefficient', ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x='Coefficient', y='Feature', data=coefficients,
                color='mediumseagreen', ax=ax)
    ax.set_title('Важность признаков (топ 15)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Значение коэффициента', fontsize=12)
    ax.set_ylabel('Признак', fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

# ==== ГЛАВНОЕ ПРИЛОЖЕНИЕ ====

def main():
    # Заголовок
    st.markdown('<p class="main-header">💳 Credit Scoring Dashboard</p>',
                unsafe_allow_html=True)
    st.markdown("---")

    # Боковая панель
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/banking.png", width=80)
        st.title("⚙️ Настройки")
        st.markdown("---")

        uploaded_file = st.file_uploader(
            "Загрузите CSV файл с данными",
            type=['csv'],
            help="Файл должен содержать данные о клиентах и целевую переменную"
        )

        st.markdown("---")
        st.markdown("### 📊 О приложении")
        st.info(
            "Это приложение для анализа кредитного скоринга "
            "и прогнозирования вероятности дефолта клиентов банка."
        )

    # Основной контент
    if uploaded_file is None:
        st.info("👆 Загрузите CSV файл для начала анализа")
        st.markdown("### 📝 Пример структуры данных")
        st.markdown("""
        Файл должен содержать следующие столбцы:
        - **SeriousDlqin2yrs**: Целевая переменная (0 или 1)
        - **age**: Возраст клиента
        - **DebtRatio**: Коэффициент задолженности
        - Другие признаки для анализа
        """)
        return

    # Загрузка данных
    df = load_data(uploaded_file)

    if df is None:
        st.error("Ошибка загрузки данных")
        return

    # Проверка наличия целевой переменной
    if 'SeriousDlqin2yrs' not in df.columns:
        st.error("В данных отсутствует целевая переменная 'SeriousDlqin2yrs'")
        return

    # Вкладки
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Обзор данных",
        "🔍 EDA",
        "🤖 Модель",
        "📈 Метрики",
        "🎯 Прогноз"
    ])

    # ==== ВКЛАДКА 1: ОБЗОР ДАННЫХ ====
    with tab1:
        st.header("📊 Обзор данных")

        col1, col2, col3, col4 = st.columns(4)
        info = get_data_info(df)

        with col1:
            st.metric("Количество строк", f"{info['Количество строк']:,}")
        with col2:
            st.metric("Количество признаков", info['Количество признаков'])
        with col3:
            st.metric("Пропущенные значения", info['Пропущенные значения'])
        with col4:
            st.metric("Дубликаты", info['Дубликаты'])

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Первые строки")
            st.dataframe(df.head(10), use_container_width=True)

        with col2:
            st.subheader("Статистика")
            st.dataframe(df.describe(), use_container_width=True)

        st.markdown("---")
        st.subheader("Типы данных")
        dtypes_df = pd.DataFrame({
            'Столбец': df.columns,
            'Тип': df.dtypes.values,
            'Пропуски': df.isnull().sum().values,
            'Уникальных': [df[col].nunique() for col in df.columns]
        })
        st.dataframe(dtypes_df, use_container_width=True)

    # ==== ВКЛАДКА 2: EDA ====
    with tab2:
        st.header("🔍 Exploratory Data Analysis")

        # Распределение целевой переменной
        st.subheader("Распределение целевой переменной")
        col1, col2 = st.columns([2, 1])

        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            counts = df['SeriousDlqin2yrs'].value_counts()
            ax.bar(['Не дефолт', 'Дефолт'], counts.values,
                   color=['skyblue', 'salmon'])
            for i, v in enumerate(counts.values):
                ax.text(i, v, f'{v:,}\n({v/len(df)*100:.1f}%)',
                        ha='center', va='bottom', fontweight='bold')
            ax.set_ylabel('Количество', fontsize=12)
            ax.set_title('Распределение дефолтов', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            st.pyplot(fig)

        with col2:
            st.markdown("### Статистика")
            st.metric("Всего клиентов", f"{len(df):,}")
            st.metric("Дефолтов", f"{counts[1]:,} ({counts[1]/len(df)*100:.1f}%)")
            st.metric("Не дефолтов", f"{counts[0]:,} ({counts[0]/len(df)*100:.1f}%)")

        st.markdown("---")

        # Распределения числовых признаков
        st.subheader("Распределения числовых признаков")
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col != 'SeriousDlqin2yrs']

        selected_feature = st.selectbox("Выберите признак", numeric_cols)

        col1, col2 = st.columns(2)

        with col1:
            fig = plot_distribution(df, selected_feature,
                                    f'Распределение: {selected_feature}')
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            df.boxplot(column=selected_feature, by='SeriousDlqin2yrs', ax=ax)
            ax.set_title(f'{selected_feature} по группам', fontsize=14, fontweight='bold')
            ax.set_xlabel('Дефолт (0 = Нет, 1 = Да)', fontsize=12)
            ax.set_ylabel(selected_feature, fontsize=12)
            plt.suptitle('')
            st.pyplot(fig)

        st.markdown("---")

        # Корреляционная матрица
        st.subheader("Корреляционная матрица")
        fig = plot_correlation_matrix(df)
        st.pyplot(fig)

    # ==== ВКЛАДКА 3: МОДЕЛЬ ====
    with tab3:
        st.header("🤖 Обучение модели")

        # Подготовка данных
        with st.spinner("Подготовка данных..."):
            missing_values, _ = check_data_quality(df)
            df_processed = preprocess_data(df, missing_values)

            # Подготовка признаков
            original_categorical = [col for col in df_processed.columns
                                    if col + '_numeric' in df_processed.columns]
            X = df_processed.drop(columns=['SeriousDlqin2yrs'] + original_categorical)
            y = df_processed['SeriousDlqin2yrs']

        st.success(f"✅ Данные подготовлены. Количество признаков: {X.shape[1]}")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Выбранные признаки")
            st.dataframe(pd.DataFrame({'Признак': X.columns}),
                         use_container_width=True, height=300)

        with col2:
            st.markdown("### Информация о разбиении")
            st.info(f"""
            - **Общее количество примеров**: {len(X):,}
            - **Обучающая выборка (75%)**: {int(len(X)*0.75):,}
            - **Тестовая выборка (25%)**: {int(len(X)*0.25):,}
            - **Стратификация**: Да
            """)

        # Обучение модели
        if st.button("🚀 Обучить модель", type="primary", use_container_width=True):
            with st.spinner("Обучение модели..."):
                model, X_train, X_test, y_train, y_test = train_model(X, y)

                # Сохранение в session_state
                st.session_state['model'] = model
                st.session_state['X_train'] = X_train
                st.session_state['X_test'] = X_test
                st.session_state['y_train'] = y_train
                st.session_state['y_test'] = y_test
                st.session_state['feature_names'] = X.columns

            st.success("✅ Модель успешно обучена!")
            st.balloons()

    # ==== ВКЛАДКА 4: МЕТРИКИ ====
    with tab4:
        st.header("📈 Метрики производительности")

        if 'model' not in st.session_state:
            st.warning("⚠️ Сначала обучите модель во вкладке 'Модель'")
        else:
            model = st.session_state['model']
            X_test = st.session_state['X_test']
            y_test = st.session_state['y_test']

            # Прогнозы
            y_pred = model.predict(X_test)
            y_pred_prob = model.predict_proba(X_test)[:, 1]

            # Метрики
            cm = confusion_matrix(y_test, y_pred)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Матрица ошибок")
                fig = plot_confusion_matrix(cm)
                st.pyplot(fig)

                tn, fp, fn, tp = cm.ravel()
                st.markdown(f"""
                - **True Negatives**: {tn:,}
                - **False Positives**: {fp:,}
                - **False Negatives**: {fn:,}
                - **True Positives**: {tp:,}
                """)

            with col2:
                st.subheader("ROC-кривая")
                fig, roc_auc = plot_roc_curve(y_test, y_pred_prob)
                st.pyplot(fig)
                st.metric("ROC AUC Score", f"{roc_auc:.4f}")

            st.markdown("---")

            # Отчет классификации
            st.subheader("Отчет классификации")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

            st.markdown("---")

            # Важность признаков
            st.subheader("Важность признаков")
            fig = plot_feature_importance(model, st.session_state['feature_names'])
            st.pyplot(fig)

    # ==== ВКЛАДКА 5: ПРОГНОЗ ====
    with tab5:
        st.header("🎯 Прогнозирование")

        if 'model' not in st.session_state:
            st.warning("⚠️ Сначала обучите модель во вкладке 'Модель'")
        else:
            model = st.session_state['model']
            feature_names = st.session_state['feature_names']

            st.markdown("### Введите данные клиента")

            # Создаем форму для ввода
            input_data = {}

            # Разбиваем на колонки
            n_features = len(feature_names)
            n_cols = 3
            cols = st.columns(n_cols)

            for idx, feature in enumerate(feature_names):
                col_idx = idx % n_cols
                with cols[col_idx]:
                    # Получаем статистику по признаку
                    feature_data = df_processed[feature]
                    min_val = float(feature_data.min())
                    max_val = float(feature_data.max())
                    mean_val = float(feature_data.mean())

                    input_data[feature] = st.number_input(
                        feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=mean_val,
                        help=f"Диапазон: {min_val:.2f} - {max_val:.2f}"
                    )

            st.markdown("---")

            if st.button("🔮 Сделать прогноз", type="primary", use_container_width=True):
                # Создаем DataFrame для прогноза
                input_df = pd.DataFrame([input_data])

                # Прогноз
                prediction = model.predict(input_df)[0]
                probability = model.predict_proba(input_df)[0]

                st.markdown("### Результат прогноза")

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Прогноз",
                              "🔴 Дефолт" if prediction == 1 else "🟢 Не дефолт")

                with col2:
                    st.metric("Вероятность дефолта", f"{probability[1]:.2%}")

                with col3:
                    st.metric("Вероятность не дефолта", f"{probability[0]:.2%}")

                # Визуализация вероятностей
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.barh(['Не дефолт', 'Дефолт'], probability,
                        color=['green', 'red'], alpha=0.7)
                ax.set_xlim([0, 1])
                ax.set_xlabel('Вероятность', fontsize=12)
                ax.set_title('Распределение вероятностей', fontsize=14, fontweight='bold')
                for i, v in enumerate(probability):
                    ax.text(v, i, f' {v:.2%}', va='center', fontweight='bold')
                ax.grid(True, alpha=0.3, axis='x')
                st.pyplot(fig)

                # Рекомендация
                st.markdown("---")
                st.markdown("### 💡 Рекомендация")
                if prediction == 1:
                    st.error(f"""
                    **Высокий риск дефолта** ({probability[1]:.2%})

                    Рекомендуется:
                    - Провести дополнительную проверку
                    - Запросить дополнительное обеспечение
                    - Рассмотреть отказ в кредите
                    """)
                else:
                    if probability[1] > 0.3:
                        st.warning(f"""
                        **Средний риск** ({probability[1]:.2%})

                        Рекомендуется:
                        - Провести стандартную проверку
                        - Рассмотреть кредит с повышенной ставкой
                        """)
                    else:
                        st.success(f"""
                        **Низкий риск дефолта** ({probability[1]:.2%})

                        Клиент является надежным заемщиком.
                        Кредит может быть одобрен на стандартных условиях.
                        """)

if __name__ == "__main__":
    main()