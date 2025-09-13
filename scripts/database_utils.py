#!/usr/bin/env python3
"""
Utilities for managing the medical consultation SQLite database.
This script provides functions to inspect, query, and manage the database.
"""

import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path


class MedicalDatabaseManager:
    """Manager class for medical consultation database operations."""
    
    def __init__(self, db_path: str = "./medical_consultations.db"):
        self.db_path = db_path
    
    def get_connection(self):
        """Get a database connection."""
        return sqlite3.connect(self.db_path)
    
    def verify_database_exists(self) -> bool:
        """Check if the database file exists."""
        return Path(self.db_path).exists()
    
    def get_table_info(self) -> Dict[str, List[str]]:
        """Get information about all tables and their columns."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        table_info = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table});")
            columns = [row[1] for row in cursor.fetchall()]
            table_info[table] = columns
        
        conn.close()
        return table_info
    
    def get_consultation_count(self) -> int:
        """Get the total number of consultations."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM patient_consultations;")
        count = cursor.fetchone()[0]
        
        conn.close()
        return count
    
    def get_completed_consultations(self) -> List[Dict[str, Any]]:
        """Get all completed consultations."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT session_id, patient_name, patient_email, 
                   consultation_start_time, consultation_end_time,
                   symptoms_reported
            FROM patient_consultations 
            WHERE completed = 1
            ORDER BY consultation_start_time DESC;
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        consultations = []
        for row in results:
            consultation = {
                "session_id": row[0],
                "patient_name": row[1],
                "patient_email": row[2],
                "start_time": row[3],
                "end_time": row[4],
                "symptoms": json.loads(row[5]) if row[5] else []
            }
            consultations.append(consultation)
        
        return consultations
    
    def get_consultation_details(self, session_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific consultation."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get consultation info
        cursor.execute("""
            SELECT * FROM patient_consultations WHERE session_id = ?;
        """, (session_id,))
        
        consultation_row = cursor.fetchone()
        if not consultation_row:
            conn.close()
            return {}
        
        # Get column names
        cursor.execute("PRAGMA table_info(patient_consultations);")
        columns = [row[1] for row in cursor.fetchall()]
        
        consultation = dict(zip(columns, consultation_row))
        
        # Get responses
        cursor.execute("""
            SELECT question_text, response_text, question_category, 
                   symptom_name, response_timestamp
            FROM patient_responses 
            WHERE session_id = ?
            ORDER BY response_timestamp;
        """, (session_id,))
        
        responses = []
        for row in cursor.fetchall():
            responses.append({
                "question": row[0],
                "response": row[1],
                "category": row[2],
                "symptom": row[3],
                "timestamp": row[4]
            })
        
        consultation["responses"] = responses
        
        # Get summary if exists
        cursor.execute("""
            SELECT summary_text, key_findings, red_flags, 
                   recommendations, next_steps
            FROM consultation_summaries 
            WHERE session_id = ?;
        """, (session_id,))
        
        summary_row = cursor.fetchone()
        if summary_row:
            consultation["summary"] = {
                "text": summary_row[0],
                "key_findings": json.loads(summary_row[1]) if summary_row[1] else [],
                "red_flags": json.loads(summary_row[2]) if summary_row[2] else [],
                "recommendations": json.loads(summary_row[3]) if summary_row[3] else [],
                "next_steps": json.loads(summary_row[4]) if summary_row[4] else []
            }
        
        conn.close()
        return consultation
    
    def export_consultation_csv(self, output_file: str = "consultations_export.csv"):
        """Export all consultations to CSV format."""
        import csv
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT pc.session_id, pc.patient_name, pc.patient_email,
                   pc.consultation_start_time, pc.consultation_end_time,
                   pc.symptoms_reported, pc.completed,
                   pr.question_text, pr.response_text, pr.question_category
            FROM patient_consultations pc
            LEFT JOIN patient_responses pr ON pc.session_id = pr.session_id
            ORDER BY pc.consultation_start_time, pr.response_timestamp;
        """)
        
        results = cursor.fetchall()
        conn.close()
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([
                'Session ID', 'Patient Name', 'Patient Email', 
                'Start Time', 'End Time', 'Symptoms', 'Completed',
                'Question', 'Response', 'Category'
            ])
            
            for row in results:
                symptoms = json.loads(row[5]) if row[5] else []
                symptoms_str = '; '.join(symptoms)
                
                writer.writerow([
                    row[0], row[1], row[2], row[3], row[4],
                    symptoms_str, row[6], row[7], row[8], row[9]
                ])
        
        print(f"Data exported to {output_file}")
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total consultations
        cursor.execute("SELECT COUNT(*) FROM patient_consultations;")
        stats['total_consultations'] = cursor.fetchone()[0]
        
        # Completed consultations
        cursor.execute("SELECT COUNT(*) FROM patient_consultations WHERE completed = 1;")
        stats['completed_consultations'] = cursor.fetchone()[0]
        
        # In-progress consultations
        cursor.execute("SELECT COUNT(*) FROM patient_consultations WHERE completed = 0;")
        stats['in_progress_consultations'] = cursor.fetchone()[0]
        
        # Total responses
        cursor.execute("SELECT COUNT(*) FROM patient_responses;")
        stats['total_responses'] = cursor.fetchone()[0]
        
        # Consultations by stage
        cursor.execute("""
            SELECT consultation_stage, COUNT(*) 
            FROM patient_consultations 
            GROUP BY consultation_stage;
        """)
        stats['consultations_by_stage'] = dict(cursor.fetchall())
        
        # Most common symptoms
        cursor.execute("""
            SELECT symptoms_reported, COUNT(*) as count
            FROM patient_consultations 
            WHERE symptoms_reported IS NOT NULL AND symptoms_reported != '[]'
            GROUP BY symptoms_reported
            ORDER BY count DESC
            LIMIT 10;
        """)
        
        common_symptoms = []
        for row in cursor.fetchall():
            try:
                symptoms = json.loads(row[0])
                common_symptoms.append({
                    'symptoms': symptoms,
                    'count': row[1]
                })
            except json.JSONDecodeError:
                continue
        
        stats['common_symptoms'] = common_symptoms
        
        # Recent activity (last 7 days)
        cursor.execute("""
            SELECT COUNT(*) 
            FROM patient_consultations 
            WHERE consultation_start_time >= datetime('now', '-7 days');
        """)
        stats['consultations_last_7_days'] = cursor.fetchone()[0]
        
        conn.close()
        return stats
    
    def cleanup_incomplete_consultations(self, older_than_hours: int = 24):
        """Clean up incomplete consultations older than specified hours."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Delete old incomplete consultations
        cursor.execute("""
            DELETE FROM patient_consultations 
            WHERE completed = 0 
            AND consultation_start_time < datetime('now', '-{} hours');
        """.format(older_than_hours))
        
        deleted_consultations = cursor.rowcount
        
        # Clean up orphaned responses
        cursor.execute("""
            DELETE FROM patient_responses 
            WHERE session_id NOT IN (
                SELECT session_id FROM patient_consultations
            );
        """)
        
        deleted_responses = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return {
            'deleted_consultations': deleted_consultations,
            'deleted_responses': deleted_responses
        }
    
    def backup_database(self, backup_path: str = None):
        """Create a backup of the database."""
        if backup_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"medical_consultations_backup_{timestamp}.db"
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
        print(f"Database backed up to {backup_path}")
        return backup_path


def main():
    """Main function for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Medical Database Management Utilities")
    parser.add_argument("--db-path", default="./medical_consultations.db", 
                       help="Path to the SQLite database file")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show database information")
    
    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export data to CSV")
    export_parser.add_argument("--output", default="consultations_export.csv",
                              help="Output CSV file path")
    
    # Consultation details command
    details_parser = subparsers.add_parser("details", help="Show consultation details")
    details_parser.add_argument("session_id", help="Session ID to show details for")
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up incomplete consultations")
    cleanup_parser.add_argument("--hours", type=int, default=24,
                               help="Clean up consultations older than this many hours")
    
    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Backup the database")
    backup_parser.add_argument("--output", help="Backup file path")
    
    # List command
    list_parser = subparsers.add_parser("list", help="List completed consultations")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    db_manager = MedicalDatabaseManager(args.db_path)
    
    if not db_manager.verify_database_exists():
        print(f"Database not found at {args.db_path}")
        print("Run the setup_medical_system.py script first to create the database.")
        return
    
    if args.command == "info":
        print(f"Database: {args.db_path}")
        print(f"Database exists: {db_manager.verify_database_exists()}")
        print("\nTable Information:")
        table_info = db_manager.get_table_info()
        for table, columns in table_info.items():
            print(f"  {table}: {', '.join(columns)}")
    
    elif args.command == "stats":
        stats = db_manager.get_database_statistics()
        print("Database Statistics:")
        print(f"  Total consultations: {stats['total_consultations']}")
        print(f"  Completed consultations: {stats['completed_consultations']}")
        print(f"  In-progress consultations: {stats['in_progress_consultations']}")
        print(f"  Total responses: {stats['total_responses']}")
        print(f"  Consultations in last 7 days: {stats['consultations_last_7_days']}")
        
        print("\nConsultations by stage:")
        for stage, count in stats['consultations_by_stage'].items():
            print(f"  {stage}: {count}")
        
        if stats['common_symptoms']:
            print("\nMost common symptom combinations:")
            for i, symptom_data in enumerate(stats['common_symptoms'][:5], 1):
                symptoms_str = ', '.join(symptom_data['symptoms'])
                print(f"  {i}. {symptoms_str} (count: {symptom_data['count']})")
    
    elif args.command == "export":
        db_manager.export_consultation_csv(args.output)
    
    elif args.command == "details":
        details = db_manager.get_consultation_details(args.session_id)
        if not details:
            print(f"No consultation found with session ID: {args.session_id}")
            return
        
        print(f"Consultation Details for {args.session_id}:")
        print(f"  Patient: {details.get('patient_name', 'N/A')}")
        print(f"  Email: {details.get('patient_email', 'N/A')}")
        print(f"  Stage: {details.get('consultation_stage', 'N/A')}")
        print(f"  Started: {details.get('consultation_start_time', 'N/A')}")
        print(f"  Completed: {details.get('completed', False)}")
        
        symptoms = json.loads(details.get('symptoms_reported', '[]'))
        if symptoms:
            print(f"  Symptoms: {', '.join(symptoms)}")
        
        print(f"\nResponses ({len(details.get('responses', []))}):")
        for resp in details.get('responses', []):
            print(f"  Q: {resp['question']}")
            print(f"  A: {resp['response']}")
            print(f"  Category: {resp['category']}")
            print()
    
    elif args.command == "cleanup":
        result = db_manager.cleanup_incomplete_consultations(args.hours)
        print(f"Cleaned up {result['deleted_consultations']} incomplete consultations")
        print(f"Cleaned up {result['deleted_responses']} orphaned responses")
    
    elif args.command == "backup":
        backup_path = db_manager.backup_database(args.output)
        print(f"Database backed up to: {backup_path}")
    
    elif args.command == "list":
        consultations = db_manager.get_completed_consultations()
        if not consultations:
            print("No completed consultations found.")
            return
        
        print(f"Completed Consultations ({len(consultations)}):")
        print("-" * 80)
        for consultation in consultations:
            print(f"Session ID: {consultation['session_id']}")
            print(f"Patient: {consultation['patient_name']}")
            print(f"Started: {consultation['start_time']}")
            symptoms_str = ', '.join(consultation['symptoms'])
            print(f"Symptoms: {symptoms_str}")
            print("-" * 40)


if __name__ == "__main__":
    main()