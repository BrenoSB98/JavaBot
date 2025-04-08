# -*- coding: utf-8 -*-
import streamlit as st

from utils.llm_config import llm
from utils.process_document import process_pdf
from utils.process_vector import load_existing_vector_store, add_to_vector_store

def main():
    
    vector_store = load_existing_vector_store()

    st.set_page_config(
        page_title='JavaBot',
        page_icon='img\\java.png',
    )

    st.image("img\\java.png", width=50)
    st.header("JavaBot, seu assistente de Java")
    
    with st.sidebar:
        st.header('Upload de arquivos 📄')
        uploaded_files = st.file_uploader(
            label='Faça o upload de arquivos PDF',
            type=['pdf'],
            accept_multiple_files=True,
        )

        if uploaded_files:
            with st.spinner('Processando dados...'):
                all_chunks = []
                for uploaded_file in uploaded_files:
                    chunks = process_pdf(file=uploaded_file)
                    all_chunks.extend(chunks)
                vector_store = add_to_vector_store(
                    chunks=all_chunks,
                    vector_store=vector_store,
                )

    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    question = st.chat_input('Faça sua pergunta!')

    if vector_store and question:
        for message in st.session_state.messages:
            st.chat_message(message.get('role')).write(message.get('content'))

        st.chat_message('user').write(question)
        st.session_state.messages.append({'role': 'user', 'content': question})

        with st.spinner('Buscando...'):
            response = llm(
                query=question,
                vector_store=vector_store,
            )

            st.chat_message('ai').write(response)
            st.session_state.messages.append({'role': 'ai', 'content': response})

if __name__ == "__main__":
    main()
    # Run the app
    # streamlit run app.py