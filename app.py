from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
import pickle
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


def main():
    load_dotenv()

    st.set_page_config(page_title="Ask me anything about Fupre")
    st.header("Ask your me anything about fupre")

    pdf = st.file_uploader("upload your pdf", type="pdf")

    if pdf is not None:
        pdf_reader = PdfReader(pdf)

        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, length_function=len
        )

        chunks = text_splitter.split_text(text=text)

        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                vectorStore = pickle.load(f)
        else:
            embeddings = OpenAIEmbeddings()
            vectorStore = FAISS.from_texts(chunks, embedding=embeddings)

            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(vectorStore, f)

        query = st.text_input("Ask questions about fupre")

        if query:
            docs = vectorStore.similarity_search(query=query, k=3)
            llm = OpenAI(temperature=0)
            chain = load_qa_chain(llm=llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(response)


if __name__ == "__main__":
    main()
