import os
import argparse
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def main():
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Minimal PDF RAG CLI")
    parser.add_argument("--ingest", type=str, help="Path to PDF file for ingestion")
    parser.add_argument("--ask", type=str, help="Question to ask about the ingested document")
    
    args = parser.parse_args()
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in environment.")
        exit(1)
        
    if args.ingest:
        # TODO
        print("provide ingest")
        
    elif args.ask:
        # TODO
        print("provide ask")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
