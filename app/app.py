import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import numpy as np

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Smart Ride-Hailing Analytics",
    page_icon="🚗",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
/* Main background */
.main { background-color: #0f1117; }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #2e3250;
}

/* Sidebar text FIX */
[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* Sidebar labels */
[data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
    font-weight: 500;
}

/* Multiselect / dropdown */
[data-testid="stSidebar"] .stMultiSelect div,
[data-testid="stSidebar"] .stSelectbox div {
    background-color: #1f2937 !important;
    color: #e5e7eb !important;
    border-radius: 8px;
}

/* Fix selected values */
[data-testid="stSidebar"] span {
    color: #e5e7eb !important;
}

/* KPI cards */
[data-testid="metric-container"] {
    background-color: #1c1f2b;
    border: 1px solid #2e3250;
    border-radius: 12px;
    padding: 16px;
}

/* Section headers */
.section-header {
    color: #7c83fd;
    font-weight: 600;
    border-bottom: 1px solid #2e3250;
    margin-bottom: 10px;
    padding-bottom: 5px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# PLOTLY THEME
# -----------------------------
PLOT_THEME = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#c5c9e0', family='monospace'),
    xaxis=dict(gridcolor='#2e3250', linecolor='#2e3250'),
    yaxis=dict(gridcolor='#2e3250', linecolor='#2e3250'),
    colorway=['#7c83fd', '#f97316', '#34d399', '#fb7185', '#facc15', '#38bdf8'],
    margin=dict(l=20, r=20, t=40, b=20)
)

def apply_theme(fig):
    fig.update_layout(**PLOT_THEME)
    return fig

# -----------------------------
# LOAD DATA & MODEL
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_path   = os.path.join(BASE_DIR, 'cleaned_data.csv')
model_path  = os.path.join(BASE_DIR, 'model', 'model.pkl')
encoder_path = os.path.join(BASE_DIR, 'model', 'encoders.pkl')

@st.cache_data
def load_data():
    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'], errors='coerce')
    df['Month']       = df['DateTime'].dt.month
    df['Day_of_Week'] = df['DateTime'].dt.day_name()
    df['Hour']        = df['DateTime'].dt.hour
    df['Is_Weekend']  = df['Day_of_Week'].isin(['Saturday', 'Sunday'])
    df['Peak'] = df['Hour'].apply(
        lambda x: 'Peak' if pd.notnull(x) and ((8 <= x <= 10) or (17 <= x <= 20)) else 'Non-Peak'
    )
    df['Is_Cancelled'] = df['Booking Status'].str.contains('Cancelled', na=False)
    return df

@st.cache_resource
def load_model():
    model    = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    return model, encoders

df_raw   = load_data()

model, encoders = load_model()


le_vehicle = encoders['vehicle']
le_pickup  = encoders['pickup']
le_drop    = encoders['drop']


# -----------------------------
# SIDEBAR FILTERS
# -----------------------------
st.sidebar.markdown("## 🎛️ Filters")
st.sidebar.markdown("---")

# Vehicle Type
vehicle_opts = df_raw['Vehicle Type'].dropna().unique().tolist()
vehicle_sel = st.sidebar.selectbox(
    "Vehicle Type",
    ["All"] + sorted(df_raw['Vehicle Type'].dropna().unique().tolist())
)





# Booking Status
status_opts = df_raw['Booking Status'].dropna().unique().tolist()
status_sel = st.sidebar.selectbox(
    "Booking Status",
    ["All"] + sorted(df_raw['Booking Status'].dropna().unique().tolist())
)

# Peak / Non-Peak
peak_opts = ['Peak', 'Non-Peak']
peak_sel = st.sidebar.selectbox(
    "Peak Period",
    ["All", "Peak", "Non-Peak"]
)

# Month Range
if df_raw['Month'].notna().any():
    month_min = int(df_raw['Month'].min())
    month_max = int(df_raw['Month'].max())
    month_range = st.sidebar.slider(
        "📅 Month Range",
        min_value=month_min,
        max_value=month_max,
        value=(month_min, month_max)
    )
else:
    month_range = (1, 12)

# Hour Range
hour_range = st.sidebar.slider("🕐 Hour Range", 0, 23, (0, 23))

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Info")
st.sidebar.info(f"Total Records: **{len(df_raw):,}**")

# -----------------------------
# APPLY FILTERS
# -----------------------------
df = df_raw.copy()
if vehicle_sel != "All":
    df = df[df['Vehicle Type'] == vehicle_sel]

if status_sel != "All":
    df = df[df['Booking Status'] == status_sel]

if peak_sel != "All":
    df = df[df['Peak'] == peak_sel]
df = df[df['Month'].between(month_range[0], month_range[1])]
df = df[df['Hour'].between(hour_range[0], hour_range[1])]

# -----------------------------
# TITLE
# -----------------------------
st.markdown("# 🚗 Smart Ride-Hailing Analytics Dashboard")
st.markdown(f"*Showing **{len(df):,}** records after filters*")
st.markdown("---")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📊 Overview", "📈 Trends & Analysis", "🤖 Prediction"])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================
with tab1:

    # --- KPI CARDS ---
    st.markdown('<p class="section-header">Key Performance Indicators</p>', unsafe_allow_html=True)

    total       = len(df)
    completed   = df[df['Booking Status'] == 'Completed'].shape[0]
    cancelled   = df[df['Is_Cancelled']].shape[0]
    revenue     = df['Booking Value'].sum()
    cancel_rate = (cancelled / total * 100) if total > 0 else 0
    avg_dist    = df['Ride Distance'].mean()
    avg_rating  = df['Customer Rating'].mean()

    row1 = st.columns(4)
    row2 = st.columns(3)

    row1[0].metric("Total Rides", f"{total:,}")
    row1[1].metric("Completed", f"{completed:,}")
    row1[2].metric("Cancelled", f"{cancelled:,}")
    row1[3].metric("Cancel Rate", f"{cancel_rate:.1f}%")

    row2[0].metric("Revenue", f"₹{revenue:,.0f}")
    row2[1].metric("Avg Distance", f"{avg_dist:.1f} km")
    row2[2].metric("Avg Rating", f"{avg_rating:.2f}")
    # c1.metric("Total Rides",       f"{total:,}")
    # c2.metric("Completed",         f"{completed:,}")
    # c3.metric("Cancelled",         f"{cancelled:,}")
    # c4.metric("Cancel Rate",       f"{cancel_rate:.1f}%")
    # c5.metric("Revenue",           f"₹{revenue:,.0f}")
    # c6.metric("Avg Distance",      f"{avg_dist:.1f} km")
    # c7.metric("Avg Cust. Rating",  f"{avg_rating:.2f} ⭐")

    st.markdown("---")

    # --- BOOKING STATUS + REVENUE BY VEHICLE ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Booking Status Distribution</p>', unsafe_allow_html=True)
        status_df = df['Booking Status'].value_counts().reset_index()
        status_df.columns = ['Status', 'Count']
        fig = px.pie(status_df, names='Status', values='Count', hole=0.45)
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">Revenue by Vehicle Type</p>', unsafe_allow_html=True)
        rev_df = df[df['Booking Status'] == 'Completed'].groupby('Vehicle Type')['Booking Value'].sum().reset_index()
        fig = px.bar(rev_df, x='Vehicle Type', y='Booking Value', color='Vehicle Type')
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # --- PAYMENT METHOD + RATING DISTRIBUTION ---
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<p class="section-header">Payment Method Distribution</p>', unsafe_allow_html=True)
        if 'Payment Method' in df.columns:
            pay_df = df[df['Booking Status'] == 'Completed'].groupby('Payment Method')['Booking Value'].sum().reset_index()
            fig = px.pie(pay_df, names='Payment Method', values='Booking Value', hole=0.45)
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown('<p class="section-header">Customer Rating Distribution</p>', unsafe_allow_html=True)
        if 'Customer Rating' in df.columns:
            fig = px.histogram(
                df[df['Booking Status'] == 'Completed'],
                x='Customer Rating', nbins=20,
                color_discrete_sequence=['#7c83fd']
            )
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # --- SUMMARY STATISTICS ---
    st.markdown('<p class="section-header">Summary Statistics</p>', unsafe_allow_html=True)
    num_cols = ['Booking Value', 'Ride Distance', 'Avg VTAT', 'Avg CTAT', 'Driver Ratings', 'Customer Rating']
    num_cols = [c for c in num_cols if c in df.columns]
    summary = df[num_cols].describe().round(2)
    st.dataframe(
        summary.style
            .background_gradient(cmap='Blues', axis=1)
            .format("{:.2f}"),
        use_container_width=True
    )

# ============================================================
# TAB 2 — TRENDS & ANALYSIS
# ============================================================
with tab2:

    # --- HOURLY DEMAND ---
    st.markdown('<p class="section-header">Hourly Demand Pattern</p>', unsafe_allow_html=True)
    hourly = df.groupby('Hour')['Booking ID'].count().reindex(range(24), fill_value=0).reset_index()
    hourly.columns = ['Hour', 'Total Rides']
    fig = px.bar(hourly, x='Hour', y='Total Rides', color_discrete_sequence=['#7c83fd'])
    fig.update_traces(marker_line_width=0)
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # --- MONTHLY TREND ---
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<p class="section-header">Monthly Trend: Rides vs Revenue</p>', unsafe_allow_html=True)
        monthly = df.groupby('Month').agg(
            Total_Rides=('Booking ID', 'count'),
            Revenue=('Booking Value', 'sum')
        ).reset_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=monthly['Month'], y=monthly['Total_Rides'],
            name='Total Rides', mode='lines+markers',
            line=dict(color='#7c83fd', width=2),
            marker=dict(size=7)
        ))
        fig.add_trace(go.Scatter(
            x=monthly['Month'], y=monthly['Revenue'],
            name='Revenue (₹)', mode='lines+markers',
            line=dict(color='#f97316', width=2),
            marker=dict(size=7),
            yaxis='y2'
        ))
        fig.update_layout(
            yaxis2=dict(overlaying='y', side='right', showgrid=False, color='#f97316'),
            legend=dict(x=0.01, y=0.99, bgcolor='rgba(0,0,0,0)'),
            **PLOT_THEME
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<p class="section-header">Weekly Demand Pattern</p>', unsafe_allow_html=True)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        week_df   = df.groupby('Day_of_Week')['Booking ID'].count().reset_index()
        week_df['Day_of_Week'] = pd.Categorical(week_df['Day_of_Week'], categories=day_order, ordered=True)
        week_df = week_df.sort_values('Day_of_Week')
        fig = px.bar(week_df, x='Day_of_Week', y='Booking ID',
                     color='Day_of_Week',
                     labels={'Booking ID': 'Total Rides'})
        fig.update_layout(showlegend=False)
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # --- CANCELLATION ANALYSIS ---
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<p class="section-header">Cancellation Rate by Vehicle Type</p>', unsafe_allow_html=True)
        cancel_veh = df.groupby('Vehicle Type').agg(
            Total=('Booking ID', 'count'),
            Cancelled=('Is_Cancelled', 'sum')
        ).reset_index()
        cancel_veh['Cancel Rate (%)'] = (cancel_veh['Cancelled'] / cancel_veh['Total'] * 100).round(2)
        fig = px.bar(cancel_veh, x='Vehicle Type', y='Cancel Rate (%)',
                     color='Cancel Rate (%)',
                     color_continuous_scale='Reds')
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col_d:
        st.markdown('<p class="section-header">Peak vs Non-Peak Revenue</p>', unsafe_allow_html=True)
        peak_df = df[df['Booking Status'] == 'Completed'].groupby('Peak').agg(
            Revenue=('Booking Value', 'sum'),
            Rides=('Booking ID', 'count')
        ).reset_index()
        fig = px.bar(peak_df, x='Peak', y='Revenue',
                     color='Peak',
                     color_discrete_sequence=['#7c83fd', '#f97316'],
                     text='Rides')
        fig.update_traces(texttemplate='%{text} rides', textposition='outside')
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # --- DEMAND vs CANCELLATION ---
    st.markdown('<p class="section-header">Demand vs Cancellation Gap by Hour</p>', unsafe_allow_html=True)
    hourly_all    = df.groupby('Hour')['Booking ID'].count()
    hourly_cancel = df[df['Is_Cancelled']].groupby('Hour')['Booking ID'].count()
    gap_df = pd.DataFrame({
        'Total Requests': hourly_all,
        'Cancellations':  hourly_cancel
    }).fillna(0).reindex(range(24), fill_value=0).reset_index()
    gap_df.columns = ['Hour', 'Total Requests', 'Cancellations']

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gap_df['Hour'], y=gap_df['Total Requests'],
                             name='Total Requests', fill='tozeroy',
                             line=dict(color='#7c83fd')))
    fig.add_trace(go.Scatter(x=gap_df['Hour'], y=gap_df['Cancellations'],
                             name='Cancellations', fill='tozeroy',
                             line=dict(color='#fb7185')))
    fig = apply_theme(fig)
    st.plotly_chart(fig, use_container_width=True)

    # --- WAIT TIME vs RATING ---
    col_e, col_f = st.columns(2)

    with col_e:
        st.markdown('<p class="section-header">Avg Wait Time vs Customer Rating</p>', unsafe_allow_html=True)
        if 'Avg VTAT' in df.columns and 'Customer Rating' in df.columns:
            vtat_rating = df[df['Booking Status'] == 'Completed'].groupby('Customer Rating')['Avg VTAT'].mean().reset_index()
            fig = px.line(vtat_rating, x='Customer Rating', y='Avg VTAT',
                          markers=True,
                          labels={'Avg VTAT': 'Avg Wait Time (min)'},
                          color_discrete_sequence=['#34d399'])
            fig = apply_theme(fig)
            st.plotly_chart(fig, use_container_width=True)

    with col_f:
        st.markdown('<p class="section-header">Distance vs Booking Value</p>', unsafe_allow_html=True)
        fig = px.scatter(
            df[df['Booking Status'] == 'Completed'].sample(min(1000, len(df))),
            x='Ride Distance', y='Booking Value',
            opacity=0.5,
            color='Vehicle Type',
            trendline='ols'
        )
        fig = apply_theme(fig)
        st.plotly_chart(fig, use_container_width=True)


# ============================================================
# TAB 3 — PREDICTION
# ============================================================
with tab3:

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<p class="section-header">Input Ride Parameters</p>', unsafe_allow_html=True)

        vtat           = st.number_input(" Avg VTAT (Vehicle Time to Arrive)", value=10.0, step=0.5)
        ctat           = st.number_input(" Avg CTAT (Customer Time to Arrive)", value=20.0, step=0.5)
        distance       = st.number_input("Ride Distance (km)", value=5.0, step=0.5)
        hour           = st.slider("🕔 Hour of Day", 0, 23, 12)
        vehicle_input  = st.selectbox(" Vehicle Type", le_vehicle.classes_)
        pickup_input   = st.selectbox(" Pickup Location", le_pickup.classes_)
        drop_input     = st.selectbox(" Drop Location", le_drop.classes_)

        predict_btn = st.button(" Predict Cancellation Risk", use_container_width=True)

    with col_right:
        st.markdown('<p class="section-header">Prediction Result</p>', unsafe_allow_html=True)

        if predict_btn:
            vehicle_enc = le_vehicle.transform([vehicle_input])[0]
            pickup_enc  = le_pickup.transform([pickup_input])[0]
            drop_enc    = le_drop.transform([drop_input])[0]

            input_df = pd.DataFrame([[
                hour, vehicle_enc, pickup_enc, drop_enc, distance, vtat, ctat
            ]], columns=['Hour', 'Vehicle Type', 'Pickup Location', 'Drop Location',
                        'Ride Distance', 'Avg VTAT', 'Avg CTAT'])

            prediction = model.predict(input_df)[0]
            proba      = model.predict_proba(input_df)[0]
            cancel_prob   = proba[1] * 100
            complete_prob = proba[0] * 100

            if prediction == 1:
                st.markdown(
                    f'<div class="pred-cancel">❌ High Cancellation Risk<br>'
                    f'<span style="font-size:1rem; font-weight:400;">Cancellation Probability: {cancel_prob:.1f}%</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="pred-complete">✅ Likely to Complete<br>'
                    f'<span style="font-size:1rem; font-weight:400;">Completion Probability: {complete_prob:.1f}%</span></div>',
                    unsafe_allow_html=True
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=cancel_prob,
                number={'suffix': '%', 'font': {'color': '#e8eaf6'}},
                title={'text': "Cancellation Risk", 'font': {'color': '#8b92b8'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#8b92b8'},
                    'bar': {'color': '#fb7185' if prediction == 1 else '#34d399'},
                    'bgcolor': '#1c1f2b',
                    'steps': [
                        {'range': [0, 40],  'color': '#0d2b1e'},
                        {'range': [40, 70], 'color': '#2d2510'},
                        {'range': [70, 100],'color': '#3b1a1a'},
                    ],
                    'threshold': {
                        'line': {'color': '#facc15', 'width': 3},
                        'thickness': 0.8,
                        'value': 50
                    }
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color='#c5c9e0'),
                              height=280, margin=dict(l=20, r=20, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👈 Fill in the ride parameters and click **Predict** to see the cancellation risk.")

    # # --- FEATURE IMPORTANCE ---
    # st.markdown("---")
    # st.markdown('<p class="section-header">Feature Importance (Model Explanation)</p>', unsafe_allow_html=True)

    # feature_names = ['Hour', 'Vehicle Type', 'Pickup Location', 'Drop Location',
    #                  'Ride Distance', 'Avg VTAT', 'Avg CTAT']
    # importances   = model.feature_importances_
    # fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    # fi_df = fi_df.sort_values('Importance', ascending=True)

    # fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h',
    #              color='Importance', color_continuous_scale='Viridis',
    #              labels={'Importance': 'Relative Importance'})
    # fig.update_layout(coloraxis_showscale=False, **PLOT_THEME)
    # st.plotly_chart(fig, use_container_width=True)

    # st.caption("Feature importance is derived from the trained Random Forest model. "
    #            "Higher values indicate features that contribute more to predicting cancellations.")