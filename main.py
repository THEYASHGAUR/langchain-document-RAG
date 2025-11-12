# test.py
# ============================================================
# LangChain + Pinecone + OpenAI Example App
# ============================================================

# ----------------------------
# Import Required Libraries
# ----------------------------
import os
import openai
import langchain
import pinecone
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain import OpenAI

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

# ----------------------------
# 1. Read PDF Documents
# ----------------------------
def read_doc(pdf_path: str):
    """Reads all PDF files from the given directory."""
    file_loader = PyPDFLoader(pdf_path)
    documents = file_loader.load()
    return documents

# Example: load PDFs from 'documents' folder
doc = read_doc('budget_speech.pdf')
print(f"Total pages loaded: {len(doc)}")

# ----------------------------
# 2. Split Documents into Chunks
# ----------------------------
def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    """Splits documents into smaller chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = text_splitter.split_documents(docs)
    return split_docs

documents = chunk_data(docs=doc)
print(f"Total document chunks: {len(documents)}")

# ----------------------------
# 3. Generate Embeddings (OpenAI)
# ----------------------------
embeddings = OpenAIEmbeddings(api_key=os.environ['OPENAI_API_KEY'])
vectors = embeddings.embed_query("How are you?")
print(f"Embedding length: {len(vectors)}")

# ----------------------------
# 4. Initialize Pinecone Vector Database
# ----------------------------
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],
    environment="gcp-starter"
)
index_name = "langchainvector"

# Create the vector index
index = Pinecone.from_documents(doc, embeddings, index_name=index_name)

# ----------------------------
# 5. Define Query Retrieval
# ----------------------------
def retrieve_query(query, k=2):
    """Retrieve similar documents from Pinecone."""
    matching_results = index.similarity_search(query, k=k)
    return matching_results

# ----------------------------
# 6. Load QA Chain with OpenAI
# ----------------------------
llm = OpenAI(model_name="text-davinci-003", temperature=0.5)
chain = load_qa_chain(llm, chain_type="stuff")

# ----------------------------
# 7. Retrieve Answers from Documents
# ----------------------------
def retrieve_answers(query):
    """Fetches relevant documents and returns an AI-generated answer."""
    doc_search = retrieve_query(query)
    print("Matched Documents:\n", doc_search)
    response = chain.run(input_documents=doc_search, question=query)
    return response

# ----------------------------
# 8. Example Usage
# ----------------------------
if __name__ == "__main__":
    query = "How much the agriculture target will be increased by how many crore?"
    answer = retrieve_answers(query)
    print("\nAnswer:\n", answer)
