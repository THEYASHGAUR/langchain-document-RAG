# test.py
# ============================================================
# LangChain + Pinecone + OpenAI Example App
# ============================================================

# ----------------------------
# Import Required Libraries
# ----------------------------
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

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
# Initialize Pinecone client
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
index_name = "langchainvector"

# Create the vector index
index = PineconeVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
    index_name=index_name
)

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
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided context to answer the question."),
    ("human", "{input}")
])
document_chain = create_stuff_documents_chain(llm, prompt)
retriever = index.as_retriever(search_kwargs={"k": 2})
chain = create_retrieval_chain(retriever, document_chain)

# ----------------------------
# 7. Retrieve Answers from Documents
# ----------------------------
def retrieve_answers(query):
    """Fetches relevant documents and returns an AI-generated answer."""
    doc_search = retrieve_query(query)
    print("Matched Documents:")
    for i, doc in enumerate(doc_search, 1):
        print(f"\nDocument {i}:")
        print("-" * 50)
        print(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
    result = chain.invoke({"input": query})
    return result.get("answer")

# ----------------------------
# 8. Example Usage
# ----------------------------
if __name__ == "__main__":
    query = "How much the agriculture target will be increased by how many crore?"
    answer = retrieve_answers(query)
    print("\nAnswer:\n", answer)
