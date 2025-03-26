import streamlit as st

st.set_page_config(
    page_title="Agentic RAG Chatbot",
    page_icon="🤖",
    layout='wide',
)

# Crea layout para el encabezado en la página principal
col1, col2 = st.columns([1, 5])

with col1:
   st.image('inline-logo-with-tagline-and-path-2048x1152.png')

with col2:
   st.header('TAG Data Challenge 2025')

with st.sidebar:
    st.success("Select a function above")

st.markdown("#### This app was developed to allow Pathways-2-Life to access an extense knowledge base of documents.")

st.markdown("#### The app uses state-of-the-art *agentic RAG*.")

st.markdown("#### Please select the function you want to use in the side bar:")

st.markdown("- ##### Chatbot: Use the chatbot to query the documents in the knowledge base.")

st.markdown("- ##### Knowledge Base: Add documents and url links to the knowledge base. Select which documents you want to use for your queries.")
    
st.markdown("#### Develoved by the GSU Data Panthers:")
st.markdown("##### Alejandro Gorricho +1 404-661-0443")
st.markdown("##### Quan Duong +1 470-854-2300")