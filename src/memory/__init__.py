
from contextlib import AbstractAsyncContextManager

from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.settings import DatabaseType, settings
from memory.mongodb import get_mongo_saver
from memory.postgres import get_postgres_saver, get_postgres_store
from memory.sqlite import get_sqlite_saver, get_sqlite_store

import sqlite3
from pathlib import Path


def initialize_medical_database():
    """Initialize SQLite database specifically for medical consultations."""
    db_path = "./medical_consultations.db"
    
    # Ensure database exists
    if not Path(db_path).exists():
        print("Creating medical consultations database...")
        conn = sqlite3.connect(db_path)
        
        # Create the medical consultation tables
        cursor = conn.cursor()
        
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
        print(f"Medical consultation database setup complete: {db_path}")
    else:
        print(f"Medical consultation database already exists: {db_path}")
    
    return db_path


def initialize_database() -> AbstractAsyncContextManager[
    AsyncSqliteSaver | AsyncPostgresSaver | AsyncMongoDBSaver
]:
    """
    Initialize the appropriate database checkpointer based on configuration.
    Returns an initialized AsyncCheckpointer instance.
    """
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_saver()
    if settings.DATABASE_TYPE == DatabaseType.MONGO:
        return get_mongo_saver()
    else:  # Default to SQLite
        return get_sqlite_saver()


def initialize_store():
    """
    Initialize the appropriate store based on configuration.
    Returns an async context manager for the initialized store.
    """
    if settings.DATABASE_TYPE == DatabaseType.POSTGRES:
        return get_postgres_store()
    # TODO: Add Mongo store - https://pypi.org/project/langgraph-store-mongodb/
    else:  # Default to SQLite
        return get_sqlite_store()


def setup_medical_system():
    """
    Complete setup for the medical system including both 
    LangGraph memory and medical consultation database.
    """
    print("Setting up medical system databases...")
    
    # Initialize the medical consultation SQLite database
    medical_db_path = initialize_medical_database()
    
    # Verify the medical database was created properly
    conn = sqlite3.connect(medical_db_path)
    cursor = conn.cursor()
    
    # Check tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    required_tables = ["patient_consultations", "patient_responses", "consultation_summaries"]
    
    missing_tables = [table for table in required_tables if table not in tables]
    if missing_tables:
        print(f"❌ Missing tables: {missing_tables}")
        raise RuntimeError(f"Medical database setup failed - missing tables: {missing_tables}")
    else:
        print("✅ All medical database tables verified")
    
    conn.close()
    
    print("✅ Medical system setup complete!")
    return medical_db_path


__all__ = ["initialize_database", "initialize_store", "initialize_medical_database", "setup_medical_system"]
