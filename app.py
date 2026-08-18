"""
HR Analytics - Job Change Prediction & Action Plan
Streamlit app built on top of the saved model artifacts (final_model.pkl, scaler.pkl, ...).

Run locally :  streamlit run app.py
"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from xgboost import XGBClassifier
import shap

st.set_page_config(page_title="HR Job-Change Analyzer", page_icon=":bar_chart:", layout="wide")

# ----------------------------------------------------------------------------
# Where are your model artifacts? (tries a few locations)
# ----------------------------------------------------------------------------
CANDIDATE_DIRS = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'saved_models'),
    os.path.dirname(os.path.abspath(__file__)),
    'D:/iti/HR Analytics_ Job Change of Data Scientists Dataset',
]

def find_dir():
    for d in CANDIDATE_DIRS:
        if os.path.exists(os.path.join(d, 'final_model.pkl')):
            return d
    raise FileNotFoundError(
        'final_model.pkl not found. Put the app next to the .pkl files '
        '(or in a saved_models/ folder) before running.')

def find_csv():
    for d in CANDIDATE_DIRS:
        p = os.path.join(d, 'aug_train.csv')
        if os.path.exists(p):
            return p
    raise FileNotFoundError('aug_train.csv not found next to the app.')

# ----------------------------------------------------------------------------
# Same preprocessing as the notebook (Part 3)
# ----------------------------------------------------------------------------
def clean_data(df):
    df = df.copy()
    df = df.drop(columns=[c for c in ['enrollee_id', 'city'] if c in df.columns])
    df['no_company_info'] = (df['company_size'].isna() & df['company_type'].isna()).astype(int)
    df['gender'] = df['gender'].fillna('Unknown')
    df['enrolled_university'] = df['enrolled_university'].fillna('unknown')
    df['education_level'] = df['education_level'].fillna('Unknown')
    df['major_discipline'] = df['major_discipline'].fillna('Unknown')
    df['company_size'] = df['company_size'].fillna('Unknown')
    df['company_type'] = df['company_type'].fillna('Unknown')
    df['experience'] = df['experience'].fillna(df['experience'].mode()[0])
    df['last_new_job'] = df['last_new_job'].fillna(df['last_new_job'].mode()[0])
    df['experience'] = df['experience'].replace({'<1': 0, '>20': 21}).astype(int)
    df['last_new_job'] = df['last_new_job'].replace({'never': 0, '>4': 5}).astype(int)
    return df

def encode_data(df):
    df = df.copy()
    education_order = {'Primary School': 1, 'High School': 2, 'Graduate': 3,
                       'Masters': 4, 'Phd': 5, 'Unknown': 0}
    size_order = {'<10': 1, '10/49': 2, '50-99': 3, '100-500': 4, '500-999': 5,
                  '1000-4999': 6, '5000-9999': 7, '10000+': 8, 'Unknown': 0}
    df['education_level'] = df['education_level'].map(education_order)
    df['company_size'] = df['company_size'].map(size_order)
    nominal_cols = ['gender', 'relevent_experience', 'enrolled_university',
                    'major_discipline', 'company_type']
    df = pd.get_dummies(df, columns=nominal_cols, drop_first=True)
    return df

def prepare_raw(df):
    """Raw DataFrame (original units, 26 columns) ready for the twin model / SHAP."""
    df = encode_data(clean_data(df))
    return df.reindex(columns=FEATURES, fill_value=False).astype(float)

# ----------------------------------------------------------------------------
# Loading everything once (cached)
# ----------------------------------------------------------------------------
@st.cache_resource
def load_all():
    d = find_dir()
    model = joblib.load(os.path.join(d, 'final_model.pkl'))
    scaler = joblib.load(os.path.join(d, 'scaler.pkl'))
    threshold = joblib.load(os.path.join(d, 'final_threshold.pkl'))
    features = joblib.load(os.path.join(d, 'feature_columns.pkl'))

    # Twin model on original units - needed only for SHAP explanations.
    # If already saved (Part 6 notebook), load it; otherwise train it once.
    twin_path = os.path.join(d, 'twin_model.pkl')
    if os.path.exists(twin_path):
        twin = joblib.load(twin_path)
    else:
        train = pd.read_csv(find_csv())
        ids = train['enrollee_id']
        train = encode_data(clean_data(train))
        train['target'] = train['target'].astype(int)
        X = train.drop(columns=['target'])[features]
        y = train['target']
        scale_pos = (y == 0).sum() / (y == 1).sum()
        twin = XGBClassifier(random_state=42, scale_pos_weight=scale_pos)
        twin.fit(X, y)

    explainer = shap.TreeExplainer(twin)
    return model, scaler, threshold, features, twin, explainer

MODEL, SCALER, THRESHOLD, FEATURES, TWIN, EXPLAINER = None, None, None, None, None, None

# ----------------------------------------------------------------------------
# Business rules (same as Part 7 of the notebook)
# ----------------------------------------------------------------------------
FRIENDLY = {
    'city_development_index': 'City development level',
    'no_company_info': 'No company info (likely unemployed)',
    'training_hours': 'Training hours completed',
    'experience': 'Years of experience',
    'last_new_job': 'Time since last job change',
    'company_size': 'Company size',
    'education_level': 'Education level',
    'gender_Male': 'Gender: Male',
    'relevent_experience_Has relevent experience': 'Relevant work experience',
    'enrolled_university_no_enrollment': 'Not enrolled in a university program',
    'major_discipline_STEM': 'Major: STEM',
    'major_discipline_Other': 'Major: Other',
    'major_discipline_Humanities': 'Major: Humanities',
    'major_discipline_No Major': 'Major: No major',
    'major_discipline_Unknown': 'Major: Unknown',
    'company_type_Pvt Ltd': 'Employer: Pvt Ltd',
    'company_type_Public Sector': 'Employer: Public sector',
    'company_type_NGO': 'Employer: NGO',
    'company_type_Funded Startup': 'Employer: Funded startup',
    'company_type_Other': 'Employer: Other type',
    'company_type_Unknown': 'Employer: Unknown',
    'gender_Other': 'Gender: Other',
}

def risk_label(p):
    return 'High' if p >= 0.6 else ('Medium' if p >= 0.4 else 'Low')

def recommended_action(proba, row):
    if risk_label(proba) == 'Medium':
        return 'Invite to advanced training / career coaching'
    if risk_label(proba) == 'High':
        if row['no_company_info'] == 1:
            return 'Personal outreach + mentorship (likely between jobs)'
        if row['company_type_Pvt Ltd'] == 1 or row['company_type_Funded Startup'] == 1:
            return 'Salary review / growth plan conversation'
        return 'Career coaching session + project opportunities'
    return 'Monitor quarterly (no action needed)'

def top_reasons(shap_row, raw_row, k=3):
    sv = np.asarray(shap_row.values)
    if sv.ndim == 2:
        sv = sv[0]
    feats = shap_row.feature_names
    parts = []
    for j in np.argsort(-np.abs(sv))[:k]:
        label = FRIENDLY.get(feats[j], feats[j])
        value = raw_row[feats[j]]
        if isinstance(value, (float, np.floating)):
            value = round(float(value), 3)
        effect = 'INCREASES risk' if sv[j] > 0 else 'lowers risk'
        parts.append(f'{label} = {value} ({effect})')
    return '; '.join(parts)

def predict_df(raw):
    """raw: prepared DataFrame (original units) -> proba, pred, risk, action, reasons."""
    scaled = SCALER.transform(raw)
    proba = MODEL.predict_proba(scaled)[:, 1]
    out = raw.copy()
    out['probability'] = proba.round(4)
    out['prediction'] = (proba >= THRESHOLD).astype(int)
    out['risk'] = out['probability'].apply(risk_label)
    out['recommended_action'] = out.apply(
        lambda r: recommended_action(r['probability'], r), axis=1)
    if len(raw) <= 10000:
        sv = EXPLAINER(raw)
        out['top_reasons'] = [top_reasons(sv[i], raw.iloc[i]) for i in range(len(raw))]
    else:
        out['top_reasons'] = '(skipped: too many rows, uncheck details)'
    return out

# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.title(':bar_chart: HR Analytics - Job Change Prediction & Action Plan')
st.caption('Predict which data-science trainees are likely to look for a new job - '
           'then get a prioritized, explainable action plan for HR.')

with st.spinner('Loading model and building the explainer (first run only)...'):
    MODEL, SCALER, THRESHOLD, FEATURES, TWIN, EXPLAINER = load_all()

tab_single, tab_batch, tab_business, tab_about = st.tabs(
    ['Predict a candidate', 'Batch upload (CSV)', 'Business impact', 'About'])

# ------------------------------- single candidate --------------------------
with tab_single:
    st.subheader('Candidate profile')
    c1, c2, c3 = st.columns(3)
    with c1:
        city_dev = st.slider('City development index', 0.0, 1.0, 0.9, 0.001)
        gender = st.selectbox('Gender', ['Male', 'Female', 'Other', 'Unknown'])
        relevant = st.selectbox('Relevant experience', ['Has relevent experience', 'No relevent experience'])
        enrolled = st.selectbox('Enrolled university', ['no_enrollment', 'Full time course', 'Part time course', 'unknown'])
    with c2:
        education = st.selectbox('Education level', ['Graduate', 'Masters', 'Phd', 'High School', 'Primary School', 'Unknown'])
        major = st.selectbox('Major discipline', ['STEM', 'Business Degree', 'Arts', 'Humanities', 'Other', 'No Major', 'Unknown'])
        experience = st.slider('Years of experience', 0, 21, 3)
        last_job = st.slider('Time since last job change', 0, 5, 0)
    with c3:
        size = st.selectbox('Company size', ['<10', '10/49', '50-99', '100-500', '500-999', '1000-4999', '5000-9999', '10000+', 'Unknown'])
        ctype = st.selectbox('Company type', ['Pvt Ltd', 'Funded Startup', 'Public Sector', 'NGO', 'Other', 'Unknown'])
        hours = st.slider('Training hours', 0, 400, 36)
        st.write('')
        st.write('')
        predict_btn = st.button('Predict', type='primary')

    if predict_btn:
        row = pd.DataFrame([{
            'city_development_index': city_dev, 'gender': gender,
            'relevent_experience': relevant, 'enrolled_university': enrolled,
            'education_level': education, 'major_discipline': major,
            'experience': experience, 'company_size': size,
            'company_type': ctype, 'last_new_job': last_job,
            'training_hours': hours,
        }])
        raw = prepare_raw(row)
        proba = MODEL.predict_proba(SCALER.transform(raw))[:, 1][0]
        pred = int(proba >= THRESHOLD)
        sv = EXPLAINER(raw)

        r1, r2, r3, r4 = st.columns(4)
        r1.metric('Probability of job-hunting', f'{proba:.1%}')
        r2.metric('Prediction', 'LOOKING for a job' if pred else 'Not looking')
        r3.metric('Risk level', risk_label(proba))
        r4.metric('Threshold used', f'{THRESHOLD}')

        st.write('#### Why? (top reasons)')
        st.write(top_reasons(sv, raw.iloc[0]).replace('; ', chr(10) + ' - '))

        st.write('#### Recommended action')
        st.info(recommended_action(proba, raw.iloc[0]))

        fig = shap.plots.waterfall(sv[0], max_display=10, show=False)
        fig = getattr(fig, 'figure', fig)   # shap >= 0.5 returns an Axes -> take its Figure
        st.pyplot(fig)

# ------------------------------- batch upload ------------------------------
with tab_batch:
    st.subheader('Upload a CSV with new candidates')
    st.caption('Columns expected (Kaggle format): enrollee_id, city_development_index, gender, '
               'relevent_experience, enrolled_university, education_level, major_discipline, '
               'experience, company_size, company_type, last_new_job, training_hours')

    uploaded = st.file_uploader('Choose a CSV file', type=['csv'])
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        with st.spinner('Scoring candidates...'):
            raw = prepare_raw(data)
            plan = predict_df(raw)
            if 'enrollee_id' in data.columns:
                plan.insert(0, 'enrollee_id', data['enrollee_id'].values)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric('Candidates scored', len(plan))
        m2.metric('High risk', int((plan['risk'] == 'High').sum()))
        m3.metric('Medium risk', int((plan['risk'] == 'Medium').sum()))
        m4.metric('Flagged (looking)', int((plan['prediction'] == 1).sum()))

        st.write('#### Top 10 candidates to act on')
        show_cols = [c for c in ['enrollee_id', 'probability', 'risk', 'recommended_action', 'top_reasons']
                     if c in plan.columns] + ['experience', 'city_development_index', 'training_hours']
        st.dataframe(plan.sort_values('probability', ascending=False)
                     .head(10)[show_cols], use_container_width=True)

        st.write('#### Full action plan')
        st.dataframe(plan, use_container_width=True)

        st.download_button('Download action plan (CSV)',
                           plan.to_csv(index=False).encode('utf-8'),
                           file_name='hr_action_plan.csv', mime='text/csv')

# ------------------------------- business impact ---------------------------
with tab_business:
    st.subheader('How much does the model save?')
    st.caption('Validation set = 20% of aug_train (stratified). Adjust the costs and see the impact.')

    b1, b2, b3 = st.columns(3)
    with b1:
        COST_LOSS = st.number_input('Cost when a trainee leaves (USD)', 1000, 100000, 15000, 1000)
    with b2:
        COST_ACTION = st.number_input('Cost of a retention action (USD)', 100, 20000, 1500, 100)
    with b3:
        st.write('')
        compute_btn = st.button('Compute impact', type='primary')

    if compute_btn:
        with st.spinner('Evaluating on the held-out validation set...'):
            train = pd.read_csv(find_csv())
            train = encode_data(clean_data(train))
            train['target'] = train['target'].astype(int)
            X = train.drop(columns=['target'])[FEATURES]
            y = train['target']
            X_tr, X_va, y_tr, y_va = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y)
            proba = TWIN.predict_proba(X_va)[:, 1]
            y_true = y_va.values

        flagged = (proba >= THRESHOLD)
        tp = flagged & (y_true == 1)
        n_flagged = flagged.sum()
        n_seekers = y_true.sum()
        p_seeker = n_seekers / len(y_true)

        cost_do_nothing = n_seekers * COST_LOSS
        tp_random = n_flagged * p_seeker
        cost_random = cost_do_nothing - (tp_random * COST_LOSS - n_flagged * COST_ACTION)
        cost_model = cost_do_nothing - (tp.sum() * COST_LOSS - n_flagged * COST_ACTION)

        compare = pd.DataFrame({
            'strategy': ['Do nothing', 'Random targeting (same budget)', 'Model targeting (XGBoost)'],
            'candidates contacted': [0, n_flagged, n_flagged],
            'savings vs do-nothing (USD)': [0, round(cost_do_nothing - cost_random),
                                            round(cost_do_nothing - cost_model)],
            'total cost (USD)': [round(cost_do_nothing), round(cost_random), round(cost_model)],
        })
        st.dataframe(compare, use_container_width=True)

        st.success(f'Model saves **{cost_do_nothing - cost_model - (cost_do_nothing - cost_random):,.0f} USD** '
                   f'more than random targeting - and **{100 * tp.sum() / n_flagged:.0f}%** of contacted '
                   f'candidates are real job-seekers (vs ~{100 * p_seeker:.0f}% picking at random).')

# ------------------------------- about ------------------------------------
with tab_about:
    st.subheader('About this project')
    st.markdown('''
- **Competition:** HR Analytics - Job Change of Data Scientists (Kaggle), ~19.2k training rows.
- **Model:** XGBoost, threshold tuned for F1 (0.6364 on validation), 5-fold CV F1 = 0.613 +/- 0.005.
- **Explainability:** SHAP twin model (same F1) - every prediction comes with reasons in plain English.
- **Business layer:** cost model comparing do-nothing / random / model targeting, plus an action plan per candidate.
- **Repository:** https://github.com/ahmedsherif022/iti_final_project
''')