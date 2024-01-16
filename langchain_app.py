import streamlit as st
import os
import zipfile
from PyPDF2 import PdfFileReader
import docx2txt
from nltk import sent_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import pinecone
from qdrant_client import QdrantClient
import openai

# Set your OpenAI API key
openai.api_key = "you api key"

# Initialize Streamlit app
st.title("LangChain Interactive App")

# File Loading
def load_data(file_path):
    _, file_extension = os.path.splitext(file_path.lower())
    
    if file_extension == '.zip':
        return load_zip(file_path)
    elif file_extension == '.pdf':
        return load_pdf(file_path)
    elif file_extension == '.docx':
        return load_docx(file_path)
    else:
        raise ValueError("Unsupported file format")

def load_zip(zip_path):
    data = {}
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            file_name = file_info.filename
            _, file_extension = os.path.splitext(file_name.lower())
            
            if file_extension == '.pdf':
                data[file_name] = load_pdf_from_zip(zip_ref, file_name)
            elif file_extension == '.docx':
                data[file_name] = load_docx_from_zip(zip_ref, file_name)
    return data

def load_pdf_from_zip(zip_ref, file_name):
    with zip_ref.open(file_name) as file:
        return load_pdf(file)

def load_docx_from_zip(zip_ref, file_name):
    with zip_ref.open(file_name) as file:
        return load_docx(file)

def load_pdf(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfFileReader(file)
            text = ""
            for page_num in range(pdf_reader.numPages):
                page = pdf_reader.getPage(page_num)
                text += page.extractText()
            return text
    except Exception as e:
        st.error(f"Error reading PDF file: {e}")
        return None

def load_docx(docx_path):
    text = docx2txt.process(docx_path)
    return text

# Text Splitting
def split_text(text):
    sentences = sent_tokenize(text)
    return sentences

def recursively_split_by_character(text, min_chunk_size=5):
    def recursive_split(chunk):
        if len(chunk) <= min_chunk_size:
            return [chunk]
        else:
            midpoint = len(chunk) // 2
            left_chunk = chunk[:midpoint]
            right_chunk = chunk[midpoint:]
            return recursive_split(left_chunk) + recursive_split(right_chunk)

    return recursive_split(text)

# Embedding Generation
def generate_embeddings_with_hf_model(text):
    tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5")

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    embeddings = torch.mean(outputs.last_hidden_state, dim=1)
    
    return embeddings.detach().numpy()

# Vector Database
class VectorDatabase:
    def __init__(self):
        self.data = {}

    def add_vector(self, key, vector):
        self.data[key] = vector

    def get_vector(self, key):
        return self.data.get(key, None)

class PineconeDatabase:
    def __init__(self, api_key, dimension):
        self.pinecone = pinecone(api_key=api_key)
        self.dimension = dimension

    def add_vector(self, key, vector):
        self.pinecone.upsert(ids=[key], vectors=[vector], dimensions=self.dimension)

    def get_vector(self, key):
        result = self.pinecone.query(queries=[key], top_k=1)
        return result[0]['data'][0]['vector'] if result else None

class QdrantDatabase:
    def __init__(self, host, port):
        self.qdrant = QdrantClient(f"http://{host}:{port}")

    def add_vector(self, key, vector):
        self.qdrant.upsert(collection_name="your_collection", records=[{"id": key, "vector": vector}])

    def get_vector(self, key):
        result = self.qdrant.search(collection_name="your_collection", queries=[{"id": key}], top_k=1)
        return result["data"][0]["vector"] if result and result["data"] else None

# Retrievers
class VectorStoreRetriever:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def retrieve(self, query_vector):
        results = {}
        for key, vector in self.vector_db.data.items():
            similarity = cosine_similarity([query_vector], [vector])[0][0]
            results[key] = similarity
        return results

class ContextualCompressionRetriever:
    def __init__(self, context_data):
        self.context_data = context_data

    def retrieve(self, query):
        results = {}

        return results

# Output using LLM (Language Model)
def generate_output(prompt, model_name='gpt-3.5-turbo-16k'):
    response = openai.Completion.create(
        engine=model_name,
        prompt=prompt,
        max_tokens=100 
    )
    return response.choices[0].text.strip()

# Streamlit Interface
def main():
    st.sidebar.title("Menu")
    app_mode = st.sidebar.radio("Choose an action", ["Home", "Load Data","Embeddings", "Retrieve (Vector Store)", 
                                                      "Retrieve (Pinecone)", "Retrieve (Qdrant)", "Generate Output", "Exit"])

    if app_mode == "Home":
        st.title("LangChain Interactive App")
        st.write("Choose an action from the sidebar menu.")

    elif app_mode == "Load Data":
        st.title("Load Data")
        file_path = st.file_uploader("Upload a file (supported formats: PDF, DOCX, ZIP)", type=["pdf", "docx", "zip"])
        if file_path:
            loaded_data = load_data(file_path)
            st.success("File Loaded Successfully!")

            for key, text in loaded_data.items():
                sentences = split_text(text)
                embeddings = generate_embeddings(sentences)
                avg_embedding = embeddings.mean(axis=0)
                vector_db.add_vector(key, avg_embedding)

    elif app_mode == "Retrieve (Vector Store)":
        st.title("Perform Retrieval (Vector Store)")

        query_key = st.text_input("Enter the key for retrieval:")

        if st.button("Retrieve"):
            query_vector = vector_db.get_vector(query_key)
            if query_vector is not None:
                vector_store_retriever = VectorStoreRetriever(vector_db)
                results_vector_store = vector_store_retriever.retrieve(query_vector)

                st.success("Retrieval Successful!")
                st.write("Results from Vector Store Retriever:")
                st.write(results_vector_store)
            else:
                st.error("Query key not found in the vector database.")

    elif app_mode == "Retrieve (Pinecone)":
        st.title("Perform Retrieval (Pinecone)")

        query_key_pinecone = st.text_input("Enter the key for retrieval:")

        if st.button("Retrieve"):
            query_vector_pinecone = pinecone_db.get_vector(query_key_pinecone)
            if query_vector_pinecone is not None:
                retriever = VectorStoreRetriever(pinecone_db)
                results_pinecone = retriever.retrieve(query_vector_pinecone)

                st.success("Retrieval Successful!")
                st.write("Results from Pinecone Retriever:")
                st.write(results_pinecone)
            else:
                st.error("Query key not found in the Pinecone database.")

    elif app_mode == "Retrieve (Qdrant)":
        st.title("Perform Retrieval (Qdrant)")

        query_key_qdrant = st.text_input("Enter the key for retrieval:")

        if st.button("Retrieve"):
            query_vector_qdrant = qdrant_db.get_vector(query_key_qdrant)
            if query_vector_qdrant is not None:
                retriever = VectorStoreRetriever(qdrant_db)
                results_qdrant = retriever.retrieve(query_vector_qdrant)

                st.success("Retrieval Successful!")
                st.write("Results from Qdrant Retriever:")
                st.write(results_qdrant)
            else:
                st.error("Query key not found in the Qdrant database.")

    elif app_mode == "Generate Output":
        st.title("Generate Output using LLM")

        user_input = st.text_input("Enter your user input:")

        if st.button("Generate Output"):
            llm_output = generate_output(user_input)
            st.success("Output Generated Successfully!")
            st.write("LLM Output:")
            st.write(llm_output)

    elif app_mode == "Exit":
        st.balloons()
        st.stop()

if __name__ == "__main__":
    main()