import streamlit as st

st.set_page_config(
    page_title="AI Course Shift Predictor",
    page_icon="🎓",
    layout="wide",
)

menu = st.sidebar.radio(
    "Navigation",
    ["🏫 Home", "🎯 Project Overview", "🧠 Model Demo", "📊 Dataset Info"]
)


if menu == "🏫 Home":
    st.markdown(
        """
        <h1 style='text-align:center; font-size:50px; font-weight:800; color:#1E3A8A;'>
            AI Course Shift Predictor 🎓
        </h1>
        <p style='text-align:center; font-size:20px; color:#374151;'>
            Predict the most suitable academic course shift using Machine Learning and real educational data.
        </p>
        """,
        unsafe_allow_html=True
    )

    st.write("")

    col1, col2 = st.columns([1,1.3])

    with col1:
        st.markdown(
            """
            <div style="display:flex; justify-content:center; padding:20px;">
                <img src="https://estuda.com/wp-content/uploads/2025/01/prouni.png"
                     style="width:280px; border-radius:15px; box-shadow:0px 4px 12px rgba(0,0,0,0.15);" />
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            ### 📌 What is this project?

            This platform uses **Machine Learning** to determine the ideal course shift
            (Morning, Afternoon, or Evening) based on course characteristics.

            The prediction model was trained using a real academic dataset and optimized
            with supervised learning techniques.
            """
        )

    st.write("---")

    st.markdown(
        """
        <h2 style='color:#1E3A8A;'>📖 What is PROUNI?</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        The **Programa Universidade para Todos (PROUNI)** is a Brazilian government initiative 
        that provides scholarships to low-income students in private higher education institutions. 
        Since its creation in 2004, PROUNI has helped **hundreds of thousands of students each year** 
        to pursue undergraduate degrees across the country.
        """
    )

    st.markdown(
        """
        <h2 style='color:#1E3A8A;'>💡Purpose of PROUNI</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "- Expand access to higher education for financially disadvantaged students\n"
        "- Encourage social mobility through educational opportunities\n"
        "- Promote diversity in private universities across Brazil"
    )

    st.info("How this platform works? 👇")

    st.markdown(
            "This platform demonstrates the potential of Machine Learning in the educational domain. "
            "It allows users to explore how course characteristics (like modality, type of scholarship, region, and year) "
            "relate to academic shifts (Morning, Afternoon, Evening), and predict the most suitable shift for a selected course."
        )
    st.write("---")

    st.markdown("### 🔮 Curious to see it in action?")
    st.markdown(
        "Click the button on the menu to go directly to the Model Demo page and try the AI-powered course shift predictor yourself!"
    )

    st.write("---")

    st.markdown("### 📊 PROUNI by the Numbers")
    st.markdown(
        "- 🎓 Over **1.5 million scholarships awarded** since inception\n"
        "- 🏫 Covers **thousands of courses** in hundreds of private institutions\n"
        "- 🌎 Supports students from all **regions of Brazil**\n"
        "- 💰 Focused on students with **low family income**"
    )

    st.write("---")

    st.markdown("### 🚀 What you can do here")
    st.markdown(
        "- 🧠 Test the AI model with real course names\n"
        "- 📊 Visualize dataset insights (coming soon)\n"
        "- ✉️ Contact the development team"
    )

    st.success("Use the menu on the left to explore the application!")


elif menu == "🎯 Project Overview":
    st.markdown(
        """
        <style>
            .about-section {
                font-size:17px;
                line-height:1.65;
                color:#1f2937;
            }
            .about-box {
                background: #f3f4f6;
                padding: 18px 22px;
                border-radius: 12px;
                border-left: 5px solid #2563eb;
                margin-bottom: 25px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.06);
            }
            .tag {
                display: inline-block;
                padding: 4px 10px;
                background: #e0f2fe;
                color: #0369a1;
                border-radius: 8px;
                margin: 5px 6px 5px 0;
                font-size: 14px;
            }
            .title {
                font-size: 28px;
                font-weight: 800;
                color:#1E3A8A;
                margin-bottom: 10px;
            }
            .sub {
                font-size: 20px;
                font-weight: 700;
                color:#1e40af;
                margin-top: 25px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<h2 class="title">About This Project</h2>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="about-section">
        
         <h3 class="sub">🐍 Technologies Used</h3>

        <div style="margin-bottom:12px;">
            <span class="tag">Python</span>
            <span class="tag">Scikit-Learn</span>
            <span class="tag">Streamlit</span>
            <span class="tag">Pandas</span>
            <span class="tag">NumPy</span>
            <span class="tag">Pickle</span>
            <span class="tag">Machine Learning</span>
        </div>

        <h2 style='color:#1E3A8A;'>🎯 Project Challenge</h2>
        <p style='color:#374151;'>
            Our challenge was to explore a rich educational dataset from PROUNI,
            containing thousands of courses and scholarship information. The goal was
            to create an AI solution capable of predicting the most suitable course shift
            (Morning, Afternoon, or Evening), understanding patterns in course allocation,
            and providing actionable insights.
        </p>

        <div class="about-box">
            <p style="margin-bottom: 8px;"><b>Main goals of this project:</b></p>
            <ul style="margin-top:0;">
                <li>Develop a functional supervised ML classification model</li>
                <li>Explore data preprocessing and feature encoding techniques</li>
                <li>Create an intuitive interface using Streamlit</li>
                <li>Make AI-powered predictions accessible to any user</li>
                <li>Apply theoretical knowledge in a practical academic scenario</li>
            </ul>
        </div>

        <h3 class="sub">📗 How We Tackled It</h3>
        <p>
        To solve this challenge, we implemented a structured Machine Learning pipeline and a user-friendly interface:
        </p>

        <div class="about-box">
            <ul style="margin-top:0;">
                <li>Data cleaning and preprocessing to handle missing or inconsistent values</li>
                <li>Encoding categorical features using <b>OneHotEncoder</b> and <b>LabelEncoder</b></li>
                <li>Training a supervised classification model using <b>XGBoost</b></li>
                <li>Balancing imbalanced classes with oversampling techniques</li>
                <li>Serializing the trained model using <b>pickle</b> for deployment in Streamlit</li>
            </ul>
            <p style="margin-top:10px;">
                The model analyzes patterns from the dataset and predicts which shift a course most likely belongs to.
            </p>
        </div>

        <h3 class="sub">🧠 How the AI Works Behind the Scenes</h3>
        <p>
            The pipeline that powers the prediction is simple yet effective:
        </p>

        <div class="about-box">
            <ul style="margin-top:0;">
                <li>Dataset preprocessing and cleaning</li>
                <li>Encoding of course names and shifts using <b>LabelEncoder</b></li>
                <li>Training of a classification model using scikit-learn</li>
                <li>Serialization of the model with pickle</li>
                <li>Real-time inference through a Streamlit interface</li>
            </ul>
            <p style="margin-top:10px;">
                The model analyzes patterns from the dataset and predicts which shift a course most likely belongs to.
            </p>
        </div>

        """,
        unsafe_allow_html=True
    )

elif menu == "🧠 Model Demo":

    st.markdown(
        """
        <style>
            .demo-title {
                font-size: 34px;
                color: #1E3A8A;
                font-weight: 800;
                margin-bottom: 10px;
            }
            .demo-sub {
                font-size: 17px;
                color: #374151;
                margin-top: -8px;
                opacity: 0.9;
            }
            .demo-box {
                background: linear-gradient(145deg, #f0f9ff, #e0e7ff, #ede9fe);
                padding: 30px;
                border-radius: 20px;
                border: 1px solid #c7d2fe;
                box-shadow: 0px 12px 32px rgba(99,102,241,0.2);
                margin-top: 30px;
            }
            .predict-btn button {
                background: linear-gradient(90deg, #2563eb, #1d4ed8);
                color: white !important;
                border-radius: 10px !important;
                height: 48px;
                font-size: 17px;
                font-weight: 600;
                transition: 0.2s;
            }
            .result-box {
                background: #eef2ff;
                border-left: 6px solid #4f46e5;
                padding: 18px 20px;
                border-radius: 12px;
                margin-top: 25px;
                font-size: 18px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<h2 class='demo-title'>Model Demonstration</h2>", unsafe_allow_html=True)
    st.markdown(
        "<p class='demo-sub'>Use the trained AI model to predict the ideal course shift based on course characteristics.</p>",
        unsafe_allow_html=True
    )

    st.markdown("<div class='demo-box'>", unsafe_allow_html=True)

    import pickle
    import pandas as pd

    with open("modelo_xgb.pkl", "rb") as f:
        data = pickle.load(f)

    modelo = data["modelo"]
    le = data["label_encoder"]
    features = data["features"]

    dados = pd.read_csv(
        "pda-prouni-2017.csv",
        sep=";",
        encoding="latin1",
        engine="python",
        on_bad_lines="skip"
    )

    for col in ["NOME_CURSO_BOLSA", "NOME_IES_BOLSA"]:
        dados[col] = dados[col].apply(lambda x: x.encode("latin1").decode("utf-8") if isinstance(x, str) else x)

    dados.rename(columns=lambda x: x.replace("ï»¿", "").strip(), inplace=True)

    st.markdown("### ✨ Select the course details below")

    col1, col2 = st.columns(2)

    with col1:
        modalidade = st.selectbox("Mode of Education", sorted(dados["MODALIDADE_ENSINO_BOLSA"].dropna().unique()))
        tipo_bolsa = st.selectbox("Scholarship Type", sorted(dados["TIPO_BOLSA"].dropna().unique()))
        curso = st.selectbox("Course Name", sorted(dados["NOME_CURSO_BOLSA"].dropna().unique()))

    with col2:
        uf = st.selectbox("State (UF)", sorted(dados["SIGLA_UF_BENEFICIARIO_BOLSA"].dropna().unique()))
        ano = st.selectbox("Scholarship Year", sorted(dados["ANO_CONCESSAO_BOLSA"].dropna().unique()))
        regiao = st.selectbox("Region", sorted(dados["REGIAO_BENEFICIARIO_BOLSA"].dropna().unique()))

    predict_clicked = st.button("🔮 Predict Shift", use_container_width=True)

    if predict_clicked:
        entrada = pd.DataFrame([{
            "MODALIDADE_ENSINO_BOLSA": modalidade,
            "TIPO_BOLSA": tipo_bolsa,
            "NOME_CURSO_BOLSA": curso,
            "SIGLA_UF_BENEFICIARIO_BOLSA": uf,
            "ANO_CONCESSAO_BOLSA": ano,
            "REGIAO_BENEFICIARIO_BOLSA": regiao
        }])

        pred = modelo.predict(entrada)
        turno = le.inverse_transform(pred)[0]

        st.markdown(
            f"""
            <div class='result-box'>
                <b>📌 Predicted Shift:</b> 
                <span style='color:#4338ca; font-weight:800;'>{turno}</span>
                <br><br>
                The AI analyzed the course characteristics and returned the most likely shift.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("</div>", unsafe_allow_html=True)


elif menu == "📊 Dataset Info":
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd

    st.markdown("## Dataset Overview 📚")

    dados = pd.read_csv(
        "pda-prouni-2017.csv",
        sep=";",
        encoding="latin1",
        engine="python",
        on_bad_lines="skip"
    )

    for col in ["NOME_CURSO_BOLSA", "NOME_IES_BOLSA"]:
        dados[col] = dados[col].apply(lambda x: x.encode("latin1").decode("utf-8") if isinstance(x, str) else x)
    dados.rename(columns=lambda x: x.replace("ï»¿", "").strip(), inplace=True)

    st.write("Compact insights from the dataset used to train the AI model.")

    col1, col2 = st.columns(2)

    if 'idade' not in dados.columns:
        dt_col_found = next((col for col in dados.columns if 'dt_nascimento_beneficiario' in col.lower()), None)
        if dt_col_found:
            dados['data_nascimento_temp'] = pd.to_datetime(dados[dt_col_found], format='%d/%m/%Y', errors='coerce')
            dados['idade'] = 2025 - dados['data_nascimento_temp'].dt.year
            dados.drop('data_nascimento_temp', axis=1, inplace=True, errors='ignore')
    
    idade = dados['idade'].dropna() if 'idade' in dados.columns else pd.Series()
    if not idade.empty:
        with col1:
            fig1, ax1 = plt.subplots(figsize=(3.5,2.5))
            ax1.hist(idade, bins=min(20, int(np.sqrt(len(idade)))), color='#4f46e5', edgecolor='black')
            ax1.set_title('Age Distribution', fontsize=11, fontweight='bold', color='#1E3A8A')
            ax1.set_xlabel('Age', fontsize=9)
            ax1.set_ylabel('Count', fontsize=9)
            ax1.tick_params(axis='both', labelsize=8)
            st.pyplot(fig1)

    col_regiao = next((c for c in dados.columns if 'regiao' in c.lower()), None)
    if col_regiao:
        cont = dados[col_regiao].fillna('Unknown').value_counts().sort_values(ascending=False)
        with col2:
            fig2, ax2 = plt.subplots(figsize=(3.5,2.5))
            ax2.bar(cont.index, cont.values, color='#2563eb')
            ax2.set_title('Beneficiaries by Region', fontsize=11, fontweight='bold', color='#1E3A8A')
            ax2.set_ylabel('Count', fontsize=9)
            ax2.set_xticklabels(cont.index, rotation=45, ha='right', fontsize=8)
            st.pyplot(fig2)

    col_raca = next((c for c in dados.columns if 'raca' in c.lower() or 'raça' in c.lower()), None)
    if col_raca:
        cont = dados[col_raca].fillna('Not Informed').value_counts()
        with col1:
            fig3, ax3 = plt.subplots(figsize=(3.5,2.5))
            cont.plot(kind='bar', ax=ax3, color='#f59e0b')
            ax3.set_title('Race / Ethnicity', fontsize=11, fontweight='bold', color='#1E3A8A')
            ax3.set_xlabel('', fontsize=9)
            ax3.set_ylabel('Count', fontsize=9)
            ax3.tick_params(axis='x', rotation=45, labelsize=8)
            st.pyplot(fig3)

    col_tipo = next((c for c in dados.columns if 'tipo' in c.lower() and 'bolsa' in c.lower()), None)
    if col_tipo:
        cont = dados[col_tipo].fillna('Unknown').value_counts()
        with col2:
            fig4, ax4 = plt.subplots(figsize=(3.5,2.5))
            ax4.pie(cont.values, labels=cont.index, autopct='%1.0f%%', startangle=140, colors=sns.color_palette("pastel"))
            ax4.set_title('Scholarship Type', fontsize=11, fontweight='bold', color='#1E3A8A')
            st.pyplot(fig4)
