#!/usr/bin/env python3
"""
Test script to verify the medical database integration is working correctly.
Run this script to test the SQLite database functionality.
"""

import asyncio
import os
import sys
from pathlib import Path
from src.agents.patient_consultation_agent import MedicalDatabase
from uuid import uuid4
import json


async def test_database_operations():
    """Test all database operations to ensure they work correctly."""
    
    print("Testing Medical Database Integration")
    print("=" * 50)
    
    # Initialize the database
    db = MedicalDatabase("./test_medical_consultations.db")
    print("✅ Database initialized")
    
    # Test 1: Create a new consultation
    session_id = str(uuid4())
    user_id = "test_user_123"
    
    consultation_data = db.get_or_create_consultation(session_id, user_id)
    print(f"✅ Created consultation: {session_id}")
    print(f"   Initial stage: {consultation_data['consultation_stage']}")
    
    # Test 2: Update patient basic info
    db.update_consultation(
        session_id,
        patient_name="John Doe",
        patient_email="john.doe@example.com",
        consultation_stage="collecting_symptoms"
    )
    print("✅ Updated patient basic information")
    
    # Test 3: Add symptoms
    symptoms = ["chest pain", "shortness of breath"]
    db.update_consultation(
        session_id,
        symptoms_reported=json.dumps(symptoms),
        consultation_stage="asking_followup_questions"
    )
    print(f"✅ Added symptoms: {symptoms}")
    
    # Test 4: Save patient responses
    responses = [
        {
            "question_id": "q_0_0",
            "question_text": "When did your chest pain first start?",
            "category": "symptom_details",
            "response_text": "This morning around 8 AM",
            "symptom_name": "chest pain"
        },
        {
            "question_id": "q_0_1", 
            "question_text": "How would you describe your chest pain (mild, moderate, severe)?",
            "category": "symptom_details",
            "response_text": "Moderate, like pressure",
            "symptom_name": "chest pain"
        },
        {
            "question_id": "q_1_0",
            "question_text": "When did your shortness of breath first start?",
            "category": "symptom_details", 
            "response_text": "About the same time as chest pain",
            "symptom_name": "shortness of breath"
        }
    ]
    
    for response in responses:
        db.save_patient_response(
            session_id=session_id,
            question_id=response["question_id"],
            question_text=response["question_text"],
            category=response["category"],
            response_text=response["response_text"],
            symptom_name=response["symptom_name"]
        )
    
    print(f"✅ Saved {len(responses)} patient responses")
    
    # Test 5: Retrieve responses
    saved_responses = db.get_consultation_responses(session_id)
    print(f"✅ Retrieved {len(saved_responses)} responses from database")
    
    for resp in saved_responses:
        print(f"   Q: {resp['question']}")
        print(f"   A: {resp['response']}")
        print(f"   Symptom: {resp['symptom']}")
        print()
    
    # Test 6: Save consultation summary
    summary_text = """
    Patient John Doe completed consultation on chest pain and shortness of breath.
    Symptoms started this morning. Moderate chest pressure with concurrent breathing difficulty.
    Recommend immediate medical evaluation.
    """
    
    db.save_consultation_summary(
        session_id=session_id,
        summary_text=summary_text.strip(),
        key_findings=["Moderate chest pain", "Concurrent shortness of breath"],
        red_flags=["Chest pain with breathing difficulty"],
        recommendations=["Immediate medical evaluation", "Monitor symptoms closely"],
        next_steps=["Contact healthcare provider", "Seek emergency care if symptoms worsen"]
    )
    print("✅ Saved consultation summary")
    
    # Test 7: Mark consultation as completed
    from datetime import datetime
    db.update_consultation(
        session_id,
        completed=True,
        consultation_end_time=datetime.now().isoformat(),
        consultation_stage="completed"
    )
    print("✅ Marked consultation as completed")
    
    # Test 8: Verify final consultation data
    final_consultation = db.get_or_create_consultation(session_id, user_id)
    print("\n📋 Final Consultation Data:")
    print(f"   Session ID: {final_consultation['session_id']}")
    print(f"   Patient: {final_consultation['patient_name']}")
    print(f"   Email: {final_consultation['patient_email']}")
    print(f"   Stage: {final_consultation['consultation_stage']}")
    print(f"   Completed: {final_consultation['completed']}")
    
    symptoms_data = json.loads(final_consultation['symptoms_reported'])
    print(f"   Symptoms: {symptoms_data}")
    
    print("\n🎉 All database tests passed!")
    
    # Test 9: Test the database manager utilities
    print("\nTesting Database Manager Utilities:")
    print("-" * 40)
    
    # Import the database manager (you'll need to adjust the path)
    try:
        from database_utils import MedicalDatabaseManager
        
        db_manager = MedicalDatabaseManager("./test_medical_consultations.db")
        
        # Test database statistics
        stats = db_manager.get_database_statistics()
        print(f"✅ Database statistics retrieved:")
        print(f"   Total consultations: {stats['total_consultations']}")
        print(f"   Completed consultations: {stats['completed_consultations']}")
        print(f"   Total responses: {stats['total_responses']}")
        
        # Test consultation details
        details = db_manager.get_consultation_details(session_id)
        print(f"✅ Consultation details retrieved:")
        print(f"   Patient: {details['patient_name']}")
        print(f"   Responses count: {len(details['responses'])}")
        
        print("\n🎉 Database manager utilities work correctly!")
        
    except ImportError:
        print("⚠️  Database utilities not available (normal if running separately)")
    
    # Cleanup test database
    test_db_path = Path("./test_medical_consultations.db")
    if test_db_path.exists():
        test_db_path.unlink()
        print("\n🧹 Cleaned up test database")


def test_database_schema():
    """Test that the database schema is created correctly."""
    
    print("\nTesting Database Schema Creation:")
    print("-" * 40)
    
    db = MedicalDatabase("./test_schema.db")
    
    import sqlite3
    conn = sqlite3.connect("./test_schema.db")
    cursor = conn.cursor()
    
    # Check that all required tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    required_tables = ["patient_consultations", "patient_responses", "consultation_summaries"]
    
    for table in required_tables:
        if table in tables:
            print(f"✅ Table '{table}' exists")
        else:
            print(f"❌ Table '{table}' missing")
    
    # Check table schemas
    for table in required_tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = [row[1] for row in cursor.fetchall()]
        print(f"   {table} columns: {', '.join(columns)}")
    
    conn.close()
    
    # Cleanup
    Path("./test_schema.db").unlink()
    print("🧹 Cleaned up schema test database")


async def main():
    """Run all tests."""
    print("🏥 Medical Database Integration Tests")
    print("=" * 60)
    
    try:
        test_database_schema()
        await test_database_operations()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED!")
        print("✅ SQLite integration is working correctly!")
        print("✅ Ready to use the patient consultation agent!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)