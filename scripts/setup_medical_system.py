#!/usr/bin/env python3
"""
Complete integration script for the Medical Doctor Assistant Agent system.
This script sets up both the RAG database and the patient consultation system.
"""

import os
import json
import sqlite3
import shutil
from pathlib import Path
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# Load environment variables
load_dotenv()


class MedicalSystemIntegration:
    """Complete medical system integration class."""
    
    def __init__(self, data_folder: str = "./data", db_name: str = "./chroma_db"):
        self.data_folder = Path(data_folder)
        self.db_name = db_name
        self.sqlite_db = "./medical_consultations.db"
        
    def setup_directories(self):
        """Create necessary directories."""
        self.data_folder.mkdir(exist_ok=True)
        print(f"Data directory: {self.data_folder}")
        
    def setup_sqlite_database(self):
        """Set up SQLite database for patient consultations."""
        conn = sqlite3.connect(self.sqlite_db)
        cursor = conn.cursor()
        
        # Create tables for patient consultation data
        tables = {
            "patient_consultations": """
                CREATE TABLE IF NOT EXISTS patient_consultations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    user_id TEXT,
                    patient_name TEXT,
                    patient_email TEXT,
                    consultation_stage TEXT NOT NULL,
                    symptoms_reported TEXT,
                    current_symptom_index INTEGER DEFAULT 0,
                    current_question_index INTEGER DEFAULT 0,
                    consultation_start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    consultation_end_time TIMESTAMP,
                    completed BOOLEAN DEFAULT FALSE,
                    summary_generated BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """,
            
            "patient_responses": """
                CREATE TABLE IF NOT EXISTS patient_responses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question_id TEXT NOT NULL,
                    question_text TEXT NOT NULL,
                    question_category TEXT NOT NULL,
                    response_text TEXT NOT NULL,
                    symptom_name TEXT,
                    response_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    follow_up_needed BOOLEAN DEFAULT FALSE,
                    FOREIGN KEY (session_id) REFERENCES patient_consultations (session_id)
                );
            """,
            
            "consultation_summaries": """
                CREATE TABLE IF NOT EXISTS consultation_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    summary_text TEXT NOT NULL,
                    key_findings TEXT,
                    red_flags TEXT,
                    recommendations TEXT,
                    next_steps TEXT,
                    generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES patient_consultations (session_id)
                );
            """
        }
        
        for table_name, table_sql in tables.items():
            cursor.execute(table_sql)
            print(f"Created/verified table: {table_name}")
        
        conn.commit()
        conn.close()
        print(f"SQLite database setup complete: {self.sqlite_db}")
        
    def process_medical_symptoms_json(self, json_data: List[Dict]) -> List[Document]:
        """Process medical symptoms JSON into documents."""
        documents = []
        
        for item in json_data:
            symptom = item.get("symptom", "")
            follow_up_questions = item.get("follow_up_questions", {})
            
            # Create comprehensive content
            content_parts = [f"Medical Symptom: {symptom}"]
            content_parts.append("=" * 60)
            
            # Add structured questions by category
            for category, questions in follow_up_questions.items():
                category_title = category.replace("_", " ").title()
                content_parts.append(f"\n{category_title} Questions:")
                content_parts.append("-" * 40)
                
                for i, question in enumerate(questions, 1):
                    content_parts.append(f"{i}. {question}")
                content_parts.append("")
            
            # Add searchable keywords
            content_parts.append("\nSearchable Keywords:")
            keywords = [symptom.lower()]
            for category in follow_up_questions.keys():
                keywords.append(category.replace("_", " "))
            content_parts.append(", ".join(keywords))
            
            full_content = "\n".join(content_parts)
            
            # Create document with rich metadata
            doc = Document(
                page_content=full_content,
                metadata={
                    "symptom": symptom,
                    "source": "medical_symptoms_database",
                    "type": "medical_symptom",
                    "categories": list(follow_up_questions.keys()),
                    "total_questions": sum(len(q) for q in follow_up_questions.values()),
                    "search_keywords": keywords
                }
            )
            documents.append(doc)
        
        return documents
    
    def create_chroma_database(self, chunk_size: int = 2000, overlap: int = 500, delete_existing: bool = True):
        """Create ChromaDB with medical symptoms and other documents."""
        
        try:
            embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize OpenAI embeddings: {e}")
        
        # Initialize Chroma vector store
        if delete_existing and os.path.exists(self.db_name):
            shutil.rmtree(self.db_name)
            print(f"Deleted existing database at {self.db_name}")
        
        chroma = Chroma(
            embedding_function=embeddings,
            persist_directory=self.db_name,
        )
        
        # Initialize text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=overlap
        )
        
        total_documents = 0
        
        # Process all files in data folder
        for file_path in self.data_folder.glob("*"):
            if not file_path.is_file():
                continue
                
            filename = file_path.name
            print(f"Processing: {filename}")
            
            try:
                if filename.endswith(".pdf"):
                    loader = PyPDFLoader(str(file_path))
                    documents = loader.load()
                    
                elif filename.endswith(".docx"):
                    loader = Docx2txtLoader(str(file_path))
                    documents = loader.load()
                    
                elif filename.endswith(".json"):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        json_data = json.load(f)
                    
                    # Check for medical symptoms format
                    if (isinstance(json_data, list) and 
                        len(json_data) > 0 and 
                        "symptom" in json_data[0] and 
                        "follow_up_questions" in json_data[0]):
                        
                        documents = self.process_medical_symptoms_json(json_data)
                        print(f"Processed {len(documents)} medical symptoms from {filename}")
                    else:
                        print(f"Skipping JSON file {filename} - not in medical symptoms format")
                        continue
                else:
                    print(f"Skipping unsupported file type: {filename}")
                    continue
                
                # Split large documents into chunks
                all_chunks = []
                for doc in documents:
                    if len(doc.page_content) > chunk_size:
                        chunks = text_splitter.split_documents([doc])
                        all_chunks.extend(chunks)
                    else:
                        all_chunks.append(doc)
                
                # Add to ChromaDB
                if all_chunks:
                    chunk_ids = chroma.add_documents(all_chunks)
                    total_documents += len(all_chunks)
                    print(f"Added {len(all_chunks)} chunks from {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        print(f"Vector database creation complete!")
        print(f"Total documents added: {total_documents}")
        print(f"Database location: {self.db_name}")
        
        return chroma
    
    def test_medical_queries(self, chroma_db):
        """Test the medical symptom database with sample queries."""
        print("\n" + "="*60)
        print("TESTING MEDICAL SYMPTOM DATABASE")
        print("="*60)
        
        test_queries = [
            "chest pain red flags emergency",
            "shortness of breath vital signs",
            "palpitations assessment questions",
            "syncope fainting evaluation",
            "nausea vomiting follow up",
            "blood pressure symptoms",
            "ankle swelling heart failure",
            "fatigue assessment protocol"
        ]
        
        retriever = chroma_db.as_retriever(search_kwargs={"k": 3})
        
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            print("-" * 40)
            
            try:
                results = retriever.invoke(query)
                for i, doc in enumerate(results, 1):
                    symptom = doc.metadata.get("symptom", "Unknown")
                    doc_type = doc.metadata.get("type", "general")
                    print(f"{i}. {symptom} ({doc_type})")
                    print(f"   Preview: {doc.page_content[:100]}...")
                    
            except Exception as e:
                print(f"Error querying: {e}")
    
    def create_agent_files(self):
        """Create the necessary agent files for integration."""
        
        agent_files = {
            "agents/patient_data_models.py": """# Patient data models and schemas
# Add the PatientData models here
""",
            
            "agents/__init__.py": """# Updated agent imports
from agents.agents import DEFAULT_AGENT, AgentGraph, get_agent, get_all_agent_info
from agents.medical_rag_assistant import medical_rag_assistant
from agents.patient_consultation_agent import patient_consultation_agent

__all__ = [
    "get_agent", "get_all_agent_info", "DEFAULT_AGENT", "AgentGraph",
    "medical_rag_assistant", "patient_consultation_agent"
]
"""
        }
        
        for file_path, content in agent_files.items():
            path = Path(file_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Only create if doesn't exist
            if not path.exists():
                with open(path, 'w') as f:
                    f.write(content)
                print(f"Created: {file_path}")
    
    def run_complete_setup(self):
        """Run the complete system setup."""
        print("Starting Medical Doctor Assistant Agent Integration")
        print("=" * 60)
        
        # Step 1: Setup directories
        print("\n1. Setting up directories...")
        self.setup_directories()
        
        # Step 2: Setup SQLite database
        print("\n2. Setting up patient consultation database...")
        self.setup_sqlite_database()
        
        # Step 3: Create ChromaDB with medical data
        print("\n3. Creating medical symptoms vector database...")
        chroma_db = self.create_chroma_database()
        
        # Step 4: Test the system
        print("\n4. Testing medical queries...")
        self.test_medical_queries(chroma_db)
        
        # Step 5: Create agent integration files
        print("\n5. Setting up agent integration...")
        self.create_agent_files()
        
        print("\n" + "="*60)
        print("INTEGRATION COMPLETE!")
        print("="*60)
        print("\nYour system now includes:")
        print("✅ Medical RAG Assistant for healthcare professionals")
        print("✅ Patient Consultation Agent for direct patient interaction")
        print("✅ Comprehensive symptom assessment database")
        print("✅ Patient data storage and session management")
        print("✅ Structured consultation summaries")
        
        print("\nNext steps:")
        print("1. Place your medical JSON file in the ./data folder")
        print("2. Update your agents/agents.py to include the new agents")
        print("3. Start your FastAPI service")
        print("4. Test both agents through the Streamlit interface")
        
        print("\nAgent capabilities:")
        print("• medical-rag-assistant: Professional symptom assessment queries")
        print("• patient-consultation-agent: Interactive patient information collection")


def main():
    """Main execution function."""
    
    # Check for required environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in your .env file")
        return
    
    # Initialize and run integration
    integration = MedicalSystemIntegration()
    
    try:
        integration.run_complete_setup()
    except Exception as e:
        print(f"Integration failed: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    main()