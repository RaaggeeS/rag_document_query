import streamlit as st
import os
from llm import *
import tempfile

load_dotenv()
st.title("RAG based document Q&A")
st.write("Want to use the power of LLMs but need privacy? Here we have!")

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    save_path = os.path.join("temp_uploads", uploaded_file.name)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        vectors = create_vector_embeddings(save_path.split("/")[0])


user_prompt = st.text_input("Enter your query:")
option = st.selectbox(
    "How should the LLM explain you the answer?",
    ("Explain in a very simple language", "Breakdown complex parts into smaller ones.")
)


if st.button("Query PDF", type="primary"):
    response = get_results(user_prompt, vectors, option)

    print(f"Response: {response}")
    st.write(response["answer"])

    # with st.expander("Document Similarity Search"):
    #     for i, doc in enumerate(response["context"]):
    #         st.write(doc.page_content)
    #         st.write("------------")

