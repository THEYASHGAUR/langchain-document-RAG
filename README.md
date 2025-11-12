# RAG Application

A question-answering system that leverages LangChain, Pinecone, and OpenAI to provide insights from budget speech documents.

## Features

- Extract and process PDF documents
- Split content into manageable chunks
- Generate embeddings using OpenAI
- Store and retrieve documents using Pinecone vector database
- Answer natural language questions about the content

## Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd rag-app
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   PINECONE_API_KEY=your_pinecone_api_key_here
   ```

## Usage

1. Place your PDF file (e.g., `budget_speech.pdf`) in the project root directory.

2. Run the main script:
   ```bash
   python main.py
   ```

3. The script will:
   - Process the PDF document
   - Create embeddings
   - Store them in Pinecone
   - Answer questions about the document

## Architecture and Flow

The following flow mirrors the actual code in `main.py`:

```mermaid
flowchart TD
    A[User Input: Query] --> B[Load .env via dotenv]
    B --> C[Load PDF using PyPDFLoader]
    C --> D[Split into chunks using RecursiveCharacterTextSplitter]
    D --> E[Initialize OpenAIEmbeddings]
    E --> F[Initialize Pinecone client]
    F --> G[Create/Populate Vector Index using PineconeVectorStore.from_documents]
    G --> H[Build Retriever: index.as_retriever(k=2)]

    %% Query-time path
    A --> H
    H --> I[Similarity Search in Pinecone]
    I --> J[Matched Documents]

    %% LLM Answering
    J --> K[ChatOpenAI (gpt-4o-mini)]
    K --> L[ChatPromptTemplate]
    L --> M[Stuff Documents Chain]
    M --> N[Retrieval Chain]
    N --> O[Generate Answer]
    O --> P[Return/Print Answer]
```

Steps
- **Load config**: `dotenv` reads `OPENAI_API_KEY` and `PINECONE_API_KEY`.
- **Load document**: `PyPDFLoader('budget_speech.pdf')` loads pages.
- **Chunking**: `RecursiveCharacterTextSplitter` creates chunks.
- **Embeddings**: `OpenAIEmbeddings` prepares vectors.
- **Vector store**: `PineconeVectorStore.from_documents` indexes chunks in Pinecone.
- **Retriever**: `index.as_retriever(k=2)` for top-k retrieval.
- **Prompt + LLM**: `ChatPromptTemplate` + `ChatOpenAI('gpt-4o-mini')` build a documents chain.
- **Retrieval chain**: `create_retrieval_chain(retriever, document_chain)` wires retrieval to generation.
- **Answering**: `retrieve_answers(query)` performs similarity search and `chain.invoke({input: query})` to return the final answer.

## Example Query

```python
query = "What are the key highlights of the budget?"
answer = retrieve_answers(query)
print(answer)
```
