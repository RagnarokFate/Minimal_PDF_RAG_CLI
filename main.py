import os
import sys
import argparse
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Minimal PDF RAG CLI")
    parser.add_argument("--ingest", type=str, help="Path to PDF file for ingestion")
    parser.add_argument("--ask", type=str, help="Question to ask about the ingested document")
    
    args = parser.parse_args()
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("Error: GOOGLE_API_KEY not found in environment. Ensure .env is set.")
        sys.exit(1)
        
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    persist_directory = "./chroma_db"
    
    if args.ingest:
        if not os.path.exists(args.ingest):
            print(f"Error: PDF file not found at '{args.ingest}'")
            sys.exit(2)
            
        print(f"Ingesting: {args.ingest}")
        
        try:
            loader = PyPDFLoader(args.ingest)
            docs = loader.load()
            
            # 200 overlap chosen to prevent splitting mathematical formulas across chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(docs)
            
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=persist_directory
            )
            vectorstore.persist()
            
            print(f"Success: Stored {len(chunks)} technical chunks in local database.")
        except Exception as e:
            print(f"Error during ingestion pipeline: {str(e)}")
            sys.exit(3)
        
    elif args.ask:
        if not os.path.exists(persist_directory):
            print("Error: Vector database not initialized. Run --ingest first.")
            sys.exit(2)
            
        print(f"Query: {args.ask}")
        
        try:
            vectorstore = Chroma(
                persist_directory=persist_directory,
                embedding_function=embeddings
            )
            
            docs = vectorstore.similarity_search(args.ask, k=5)
            context = "\n\n".join([doc.page_content for doc in docs])
            
            llm = ChatGoogleGenerativeAI(model="models/gemini-flash-latest", temperature=0)
            
            prompt = f"""Use the following pieces of context to answer the question.
If the context doesn't contain the answer, state that you don't know based on the provided document.

Context:
{context}

Question: {args.ask}
Answer:"""
            
            response = llm.invoke(prompt)
            print(f"\n[Response]\n{response.content}")
        except Exception as e:
            print(f"Error during retrieval/generation: {str(e)}")
            sys.exit(3)
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
