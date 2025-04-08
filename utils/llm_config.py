import streamlit as st

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

MODEL = 'codellama:13b'  # Exemplo: 'llama2-7b-chat-hf'

def llm(query, vector_store):
    llm = OllamaLLM(model=MODEL)
    retriever = vector_store.as_retriever()

    system_prompt = '''
        Você é um desenvolvedor de software altamente experiente, especializado em Java, SQL, Spring Boot, JavaScript e Angular.
        Responda às perguntas com base no contexto fornecido. 
        Caso a resposta não possa ser extraída do contexto, informe claramente que não há informações disponíveis.
        Utilize a linguagem de programação solicitada na pergunta. 
        Se nenhuma linguagem for especificada, utilize Java como padrão.
        Ao gerar códigos, priorize clareza, concisão e boas práticas de programação. 
        Evite explicações desnecessárias e não inclua informações pessoais ou confidenciais.
        Contexto: {context}
    '''

    messages = [('system', system_prompt)]
    for message in st.session_state.messages:
        messages.append((message.get('role'), message.get('content')))
    messages.append(('human', '{input}'))

    prompt = ChatPromptTemplate.from_messages(messages)

    question_answer_chain = create_stuff_documents_chain(
        llm=llm,
        prompt=prompt,
    )
    chain = create_retrieval_chain(
        retriever=retriever,
        combine_docs_chain=question_answer_chain,
    )
    response = chain.invoke({'input': query})
    return response.get('answer')