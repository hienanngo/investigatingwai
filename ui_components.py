# ui_components.py
import streamlit as st

def filter_sidebar(df):
    st.sidebar.header("🔍 Filter Institutions")

    state = st.sidebar.selectbox("State", ["All"] + sorted(df["State abbreviation"].dropna().unique()))
    control = st.sidebar.selectbox("Institution Control", ["All"] + sorted(df["Public/Private"].dropna().unique()))
    degree = st.sidebar.selectbox("Degree Type", ["All"] + sorted(df["Institutional category"].dropna().unique()))
    urban = st.sidebar.selectbox("Urbanicity", ["All"] + sorted(df["Degree of urbanization (Urban-centric locale)"].dropna().unique()))

    return state, control, degree, urban
