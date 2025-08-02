import streamlit as st
from main import Model
import pandas as pd


tab1, tab2 = st.tabs(["Fun", "Empirical results"], )

with tab1:
    st.title("Dear Future Arshia!")

    st.header("""
    It's 2050, and you became a history prompt engineering professor. After telling your students how "back in the day" you used to write your own essays, you decide to finally give them an assignment:
    """, )

    st.header(""" "Prompt Gemini to write about AI revolution in 20's and 30's" """)

    st.header("""
            
    You being an expert in the field have obtained your PhD titled "Dialectics of Dialogue: Advancing Prompt Engineering Methodologies with Gemini AI Models", however shortly after your graduation, Mistral AI had a breakthrough and students started using that instead...
    """)


    st.header("""
    Seeing your career at danger, I didn't have any choice but to make a tool for you. 
    """)


    st.subheader("Enter student's LLM (Gemini/Mistral) created text")
    inp = st.text_area("",placeholder="Paste the text here")
    model = Model()

    show = False
    if inp != "":
        result, certainty = model.predict(inp)
        show = True

    if show: 
        print("Gemini" if result else "Mistral")
        if result:
            st.markdown("<h1 style='text-align: center; color: green;'>Gemini!</h1>", unsafe_allow_html=True)
        else:
            st.markdown("<h1 style='text-align: center; color: red;'>Mistral!</h1>", unsafe_allow_html=True)
        print(certainty)
        # st.slider("label", min_value=0, max_value=100, disabled=True)

    st.divider()

    st.info("""This Statistical model has been "trained" on Gemini-flash-2.0 and Mistral Large 2 (24.11) and it's very model dependant, if you want to try it out you can go to following links and paste the response back: """)

    # c1, c2 = st.columns(2)
    st.link_button("Gemini", "https://console.cloud.google.com/vertex-ai/studio/multimodal?model=gemini-2.0-flash-live-preview-04-09&inv=1&invt=Ab4W9w&project=nodal-reserve-397800")

    st.link_button("Mistral", "https://chat.mistral.ai/chat")

with tab2:

    df = pd.read_csv("./results.csv", index_col=0)
    st.markdown("<h2 style='text-align: center'>Changes in performance for different word counts</h1>", unsafe_allow_html=True)
    st.line_chart(df, y_label="Percentage", x_label="Number of words in the train dataset",)


