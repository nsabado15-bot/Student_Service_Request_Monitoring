import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Student Service Request Monitoring",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Service Request Monitoring System")

st.sidebar.title("Navigation")

menu = st.sidebar.radio(
    "Select Module",
    [
        "Home",
        "Dataset",
        "Data Analysis",
        "Prediction",
        "Dashboard",
        "System Design"
    ]
)

if "dataset" not in st.session_state:
    st.session_state.dataset = None

if "model" not in st.session_state:
    st.session_state.model = None

if "features" not in st.session_state:
    st.session_state.features = None

if menu == "Home":

    st.header("Welcome!")

    st.write(
        "This system helps student services staff analyze "
        "requests and identify possible SLA breaches."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("System Modules", "6")
    col2.metric("Prediction Target", "SLA_Breach")
    col3.metric("Decision Support", "Human Review")

    st.info(
        "This demo supports staff decisions. "
        "It does not automatically deny student services."
    )

elif menu == "Dataset":

    st.header("Dataset Upload and Validation")

    uploaded_file = st.file_uploader(
        "Upload your student service requests CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:
            df = pd.read_csv(uploaded_file)

            if df.empty:
                st.error("The uploaded CSV is empty.")

            else:
                st.session_state.dataset = df
                st.session_state.model = None

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Rows", len(df))
                col2.metric("Total Columns", len(df.columns))
                col3.metric(
                    "Duplicate Rows",
                    int(df.duplicated().sum())
                )

                st.subheader("Dataset Preview")
                st.dataframe(df.head(20), use_container_width=True)

                st.subheader("Missing Values")
                missing = df.isnull().sum().rename("Missing Values")
                st.dataframe(missing)

                st.subheader("Column Names")
                st.write(list(df.columns))

        except Exception as error:
            st.error(f"Unable to read CSV: {error}")

    elif st.session_state.dataset is not None:

        st.success("Dataset is loaded.")

        st.dataframe(
            st.session_state.dataset.head(20),
            use_container_width=True
        )

    else:
        st.info("Upload a CSV file to begin.")

elif menu == "Data Analysis":

    st.header("Exploratory Data Analysis")

    df = st.session_state.dataset

    if df is None:

        st.warning("Upload a CSV in the Dataset module first.")

    else:

        st.subheader("1. Descriptive Statistics")

        st.dataframe(df.describe(include="all").T)

        st.subheader("2. Category Distribution")

        categorical = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        if categorical:

            category = st.selectbox(
                "Choose a category",
                categorical
            )

            counts = df[category].astype(str).value_counts()

            st.bar_chart(counts)

        st.subheader("3. Numerical Relationships")

        numerical = df.select_dtypes(
            include="number"
        ).columns.tolist()

        if len(numerical) >= 2:

            x_axis = st.selectbox(
                "Choose X-axis",
                numerical,
                key="analysis_x"
            )

            y_axis = st.selectbox(
                "Choose Y-axis",
                numerical,
                key="analysis_y"
            )

            st.scatter_chart(df, x=x_axis, y=y_axis)

        elif numerical:

            st.bar_chart(df[numerical[0]])

elif menu == "Prediction":

    st.header("SLA Breach Prediction")

    df = st.session_state.dataset

    if df is None:

        st.warning("Upload a CSV in the Dataset module first.")

    elif "SLA_Breach" not in df.columns:

        st.error(
            "The dataset must contain the SLA_Breach target column."
        )

    else:

        st.warning(
            "DEMO MODEL: This trains a new model using the uploaded "
            "sample dataset. It is not your original midterm model."
        )

        if st.button("Train Demo Model"):

            try:
                data = df.dropna(subset=["SLA_Breach"]).copy()

                y = data["SLA_Breach"].astype(str).str.strip()

                X = data.drop(
                    columns=["SLA_Breach", "Request_ID"],
                    errors="ignore"
                )

                valid_rows = y.isin(["Yes", "No"])
                X = X.loc[valid_rows]
                y = y.loc[valid_rows]

                if len(X) < 10:
                    st.error(
                        "At least 10 labeled records are recommended."
                    )

                elif y.nunique() < 2:
                    st.error(
                        "The target must contain both Yes and No labels."
                    )

                elif X.shape[1] == 0:
                    st.error("No input features were found.")

                else:
                    categorical_features = X.select_dtypes(
                        include=["object", "category", "bool"]
                    ).columns.tolist()

                    numerical_features = X.select_dtypes(
                        include="number"
                    ).columns.tolist()

                    transformers = []

                    if numerical_features:
                        numeric_pipeline = Pipeline([
                            (
                                "imputer",
                                SimpleImputer(strategy="median")
                            )
                        ])

                        transformers.append((
                            "numeric",
                            numeric_pipeline,
                            numerical_features
                        ))

                    if categorical_features:
                        categorical_pipeline = Pipeline([
                            (
                                "imputer",
                                SimpleImputer(
                                    strategy="most_frequent"
                                )
                            ),
                            (
                                "encoder",
                                OneHotEncoder(
                                    handle_unknown="ignore"
                                )
                            )
                        ])

                        transformers.append((
                            "categorical",
                            categorical_pipeline,
                            categorical_features
                        ))

                    preprocessing = ColumnTransformer(
                        transformers=transformers
                    )

                    model = Pipeline([
                        ("preprocessing", preprocessing),
                        (
                            "classifier",
                            RandomForestClassifier(
                                n_estimators=100,
                                random_state=42,
                                class_weight="balanced"
                            )
                        )
                    ])

                    model.fit(X, y)

                    st.session_state.model = model
                    st.session_state.features = X.columns.tolist()

                    st.success("Demo model trained successfully!")

                    st.write("Training records:", len(X))
                    st.write("Input features:", list(X.columns))

            except Exception as error:
                st.error(f"Model training failed: {error}")

        if st.session_state.model is not None:

            st.subheader("Predict a Student Service Request")

            model = st.session_state.model
            features = st.session_state.features

            inputs = {}

            for feature in features:

                if feature not in df.columns:
                    continue

                column = df[feature]

                if (
                    pd.api.types.is_numeric_dtype(column)
                    and column.notna().any()
                ):

                    default_value = float(column.median())

                    inputs[feature] = st.number_input(
                        feature,
                        value=default_value
                    )

                else:

                    options = column.dropna().astype(str).unique().tolist()

                    if options:
                        inputs[feature] = st.selectbox(
                            feature,
                            options
                        )
                    else:
                        inputs[feature] = st.text_input(feature)

            if st.button("Predict SLA Breach"):

                try:
                    input_df = pd.DataFrame([inputs])
                    input_df = input_df[features]

                    prediction = model.predict(input_df)[0]

                    st.subheader("Prediction Result")

                    if str(prediction).lower() == "yes":
                        st.error(
                            "⚠️ The demo model predicts a possible SLA breach."
                        )
                    else:
                        st.success(
                            "✅ The demo model predicts no SLA breach."
                        )

                    if hasattr(model, "predict_proba"):
                        probabilities = model.predict_proba(input_df)[0]

                        st.write("Model probability estimates:")

                        probability_df = pd.DataFrame({
                            "Class": model.classes_,
                            "Probability": probabilities
                        })

                        st.dataframe(
                            probability_df,
                            use_container_width=True
                        )

                    st.info(
                        "This is a demonstration prediction, not a "
                        "verified operational result. Staff must review "
                        "the request before making decisions."
                    )

                except Exception as error:
                    st.error(f"Prediction failed: {error}")

elif menu == "Dashboard":

    st.header("Decision-Support Dashboard")

    df = st.session_state.dataset

    if df is None:

        st.warning("Upload a CSV in the Dataset module first.")

    else:

        col1, col2 = st.columns(2)

        col1.metric("Total Requests", len(df))

        if "SLA_Breach" in df.columns:

            breach_count = (
                df["SLA_Breach"].astype(str).str.strip().str.lower()
                == "yes"
            ).sum()

            col2.metric("Historical SLA Breaches", int(breach_count))

            st.subheader("Historical SLA Breach Distribution")

            st.bar_chart(
                df["SLA_Breach"].astype(str).value_counts()
            )

            st.caption(
                "These counts describe historical labels, not new "
                "model predictions."
            )

        st.subheader("Request Summary")

        if "Service_Type" in df.columns:
            st.bar_chart(df["Service_Type"].value_counts())

        st.info(
            "Use the dashboard to support staff review, prioritization, "
            "and follow-up."
        )

elif menu == "System Design":

    st.header("System Design")

    st.subheader("Actors")

    st.write(
        "- Student Services Staff\n"
        "- System Administrator\n"
        "- Authorized Decision-Maker"
    )

    st.subheader("Functional Requirements")

    st.write(
        "1. Upload and validate CSV files.\n\n"
        "2. Display data quality summaries.\n\n"
        "3. Display exploratory data analysis charts.\n\n"
        "4. Train a demo model and predict possible SLA breaches.\n\n"
        "5. Present historical information for staff review.\n\n"
        "6. Support human decision-making."
    )

    st.subheader("Decision Workflow")

    st.write(
        "Upload Dataset → Validate Data → Analyze Requests → "
        "Predict SLA Risk → Staff Review → Authorized Decision"
    )

    st.subheader("Important Limitation")

    st.write(
        "The current model is trained from the uploaded demonstration "
        "dataset. It must be replaced with the original midterm model "
        "and evaluated properly before operational use."
    )