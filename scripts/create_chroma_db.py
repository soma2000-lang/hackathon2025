import os
import json
import shutil

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Load environment variables from the .env file
load_dotenv()


def process_medical_symptoms_json(json_data):
    """
    Convert medical symptom JSON data into searchable documents.
    Each symptom becomes a separate document with all follow-up questions.
    """
    documents = []
    
    for item in json_data:
        symptom = item.get("symptom", "")
        follow_up_questions = item.get("follow_up_questions", {})
        
        # Create main content for the symptom
        content_parts = [f"Medical Symptom: {symptom}"]
        content_parts.append("=" * 50)
        
        # Add all follow-up questions organized by category
        for category, questions in follow_up_questions.items():
            category_title = category.replace("_", " ").title()
            content_parts.append(f"\n{category_title} Questions:")
            content_parts.append("-" * 30)
            for i, question in enumerate(questions, 1):
                content_parts.append(f"{i}. {question}")
            content_parts.append("")
        
        # Combine all content
        full_content = "\n".join(content_parts)
        
        # Create document with rich metadata
        doc = Document(
            page_content=full_content,
            metadata={
                "symptom": symptom,
                "source": "medical_symptoms_database",
                "type": "medical_symptom",
                "categories": list(follow_up_questions.keys()),
                "total_questions": sum(len(questions) for questions in follow_up_questions.values())
            }
        )
        documents.append(doc)
    
    return documents


def create_chroma_db(
    folder_path: str,
    db_name: str = "./chroma_db",
    delete_chroma_db: bool = True,
    chunk_size: int = 2000,
    overlap: int = 500,
):
    embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

    # Initialize Chroma vector store
    if delete_chroma_db and os.path.exists(db_name):
        shutil.rmtree(db_name)
        print(f"Deleted existing database at {db_name}")

    chroma = Chroma(
        embedding_function=embeddings,
        persist_directory=f"./{db_name}",
    )

    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)

    # Iterate over files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        try:
            # Load document based on file extension
            if filename.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
            elif filename.endswith(".docx"):
                loader = Docx2txtLoader(file_path)
                documents = loader.load()
                
            elif filename.endswith(".json"):
                # Handle JSON files
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                
                # Check if it's the medical symptoms format
                if (isinstance(json_data, list) and 
                    len(json_data) > 0 and 
                    "symptom" in json_data[0] and 
                    "follow_up_questions" in json_data[0]):
                    
                    # Use custom processing for medical symptoms
                    documents = process_medical_symptoms_json(json_data)
                    print(f"Processed {len(documents)} medical symptoms from {filename}")
                else:
                    # Skip other JSON formats for now
                    print(f"Skipping JSON file {filename} - not in medical symptoms format")
                    continue
            else:
                print(f"Skipping unsupported file type: {filename}")
                continue
            
            # Split documents into chunks (if they're large)
            all_chunks = []
            for doc in documents:
                if len(doc.page_content) > chunk_size:
                    chunks = text_splitter.split_documents([doc])
                    all_chunks.extend(chunks)
                else:
                    all_chunks.append(doc)
            
            # Add chunks to Chroma vector store
            if all_chunks:
                chunk_ids = chroma.add_documents(all_chunks)
                print(f"Added {len(all_chunks)} chunks from {filename}")
            else:
                print(f"No content found in {filename}")
                
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue

    print(f"Vector database created and saved in {db_name}.")
    return chroma


def query_medical_symptoms(chroma_db, query: str, k: int = 3):
    """
    Query the database for medical symptoms and return formatted results.
    """
    retriever = chroma_db.as_retriever(search_kwargs={"k": k})
    similar_docs = retriever.invoke(query)
    
    results = []
    for i, doc in enumerate(similar_docs, start=1):
        result = {
            "rank": i,
            "symptom": doc.metadata.get("symptom", "Unknown"),
            "content": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
            "categories": doc.metadata.get("categories", []),
            "relevance_score": getattr(doc, 'relevance_score', None)
        }
        results.append(result)
    
    return results


if __name__ == "__main__":
    # Path to the folder containing the documents (including your JSON file)
    folder_path = "./data"  # Make sure your Datasetab94d2b.json is in this folder

    # Create the Chroma database
    chroma = create_chroma_db(folder_path=folder_path)

    # Test medical symptom queries
    test_queries = [
        "chest pain questions",
        "shortness of breath assessment",
        "palpitations with dizziness",
        "fainting episode evaluation",
        "blood pressure symptoms"
    ]

    print("\n" + "="*60)
    print("TESTING MEDICAL SYMPTOM QUERIES")
    print("="*60)

    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 40)
        
        results = query_medical_symptoms(chroma, query, k=2)
        
        for result in results:
            print(f"\n📋 Rank {result['rank']}: {result['symptom']}")
            print(f"📂 Categories: {', '.join(result['categories'])}")
            print(f"📄 Content preview: {result['content'][:200]}...")
            print()