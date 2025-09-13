import json
import re
import sqlite3
from datetime import datetime
from typing import Literal, Dict, List, Any
from enum import Enum
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnableSerializable,
)
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.managed import RemainingSteps
from langgraph.prebuilt import ToolNode
from langgraph.store.base import BaseStore

from agents.llama_guard import LlamaGuard, LlamaGuardOutput, SafetyAssessment
from agents.tools import database_search, get_symptom_followup_questions
from core import get_model, settings


class ConsultationStage(Enum):
    GREETING = "greeting"
    COLLECTING_BASIC_INFO = "collecting_basic_info"
    COLLECTING_SYMPTOMS = "collecting_symptoms"
    ASKING_FOLLOWUP_QUESTIONS = "asking_followup_questions"
    SUMMARY = "summary"
    COMPLETED = "completed"


class PatientConsultationState(MessagesState, total=False):
    """State for patient consultation agent."""
    
    safety: LlamaGuardOutput
    remaining_steps: RemainingSteps
    consultation_stage: ConsultationStage
    patient_data: Dict[str, Any]
    current_symptom_index: int
    current_question_index: int
    symptom_questions: List[Dict[str, Any]]
    consultation_complete: bool
    session_id: str


tools = [database_search, get_symptom_followup_questions]


class MedicalDatabase:
    """Helper class to manage SQLite database operations for medical consultations."""
    
    def __init__(self, db_path: str = "./medical_consultations.db"):
        self.db_path = db_path
        self.ensure_database_exists()
    
    def ensure_database_exists(self):
        """Ensure the medical consultation database exists with proper tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables if they don't exist
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
        
        conn.commit()
        conn.close()
    
    def get_or_create_consultation(self, session_id: str, user_id: str = None) -> Dict[str, Any]:
        """Get existing consultation or create a new one."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Try to get existing consultation
        cursor.execute(
            "SELECT * FROM patient_consultations WHERE session_id = ?",
            (session_id,)
        )
        result = cursor.fetchone()
        
        if result:
            # Return existing consultation data
            columns = [desc[0] for desc in cursor.description]
            consultation_data = dict(zip(columns, result))
            conn.close()
            return consultation_data
        else:
            # Create new consultation
            cursor.execute("""
                INSERT INTO patient_consultations 
                (session_id, user_id, consultation_stage, symptoms_reported)
                VALUES (?, ?, ?, ?)
            """, (session_id, user_id, ConsultationStage.GREETING.value, "[]"))
            
            conn.commit()
            conn.close()
            
            return {
                "session_id": session_id,
                "user_id": user_id,
                "consultation_stage": ConsultationStage.GREETING.value,
                "patient_name": None,
                "patient_email": None,
                "symptoms_reported": "[]",
                "current_symptom_index": 0,
                "current_question_index": 0,
                "completed": False
            }
    
    def update_consultation(self, session_id: str, **kwargs):
        """Update consultation data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Build dynamic update query
        set_clauses = []
        values = []
        for key, value in kwargs.items():
            if key in ['patient_name', 'patient_email', 'consultation_stage', 
                      'symptoms_reported', 'current_symptom_index', 
                      'current_question_index', 'completed', 'consultation_end_time']:
                set_clauses.append(f"{key} = ?")
                values.append(value)
        
        if set_clauses:
            set_clauses.append("updated_at = CURRENT_TIMESTAMP")
            query = f"UPDATE patient_consultations SET {', '.join(set_clauses)} WHERE session_id = ?"
            values.append(session_id)
            cursor.execute(query, values)
            conn.commit()
        
        conn.close()
    
    def save_patient_response(self, session_id: str, question_id: str, 
                            question_text: str, category: str, response_text: str,
                            symptom_name: str = None):
        """Save a patient response to a question."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO patient_responses 
            (session_id, question_id, question_text, question_category, 
             response_text, symptom_name)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, question_id, question_text, category, response_text, symptom_name))
        
        conn.commit()
        conn.close()
    
    def save_consultation_summary(self, session_id: str, summary_text: str,
                                 key_findings: List[str] = None, 
                                 red_flags: List[str] = None,
                                 recommendations: List[str] = None,
                                 next_steps: List[str] = None):
        """Save the final consultation summary."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO consultation_summaries 
            (session_id, summary_text, key_findings, red_flags, recommendations, next_steps)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session_id, summary_text, 
              json.dumps(key_findings or []), 
              json.dumps(red_flags or []),
              json.dumps(recommendations or []),
              json.dumps(next_steps or [])))
        
        conn.commit()
        conn.close()
    
    def get_consultation_responses(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all responses for a consultation session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT question_text, response_text, question_category, symptom_name
            FROM patient_responses 
            WHERE session_id = ?
            ORDER BY response_timestamp
        """, (session_id,))
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                "question": row[0],
                "response": row[1], 
                "category": row[2],
                "symptom": row[3]
            }
            for row in results
        ]


def create_patient_consultation_prompt() -> str:
    current_date = datetime.now().strftime("%B %d, %Y")
    return f"""
    You are PatientBot, a compassionate medical consultation assistant. Today's date is {current_date}.

    IMPORTANT: You are NOT a general chatbot. You have ONE specific job - collect patient information systematically.

    CONSULTATION STAGES:
    1. GREETING: Welcome and explain the process
    2. COLLECTING_BASIC_INFO: Get name and email only
    3. COLLECTING_SYMPTOMS: Get symptom description  
    4. ASKING_FOLLOWUP_QUESTIONS: Ask specific medical questions
    5. SUMMARY: Provide final summary
    6. COMPLETED: End consultation

    STRICT RULES:
    - Ask ONLY ONE question at a time
    - Wait for patient response before proceeding
    - Do NOT repeat the same question
    - Progress through stages sequentially
    - Keep responses short and focused
    
    COMMUNICATION STYLE:
    - Be warm but professional
    - Use simple, patient-friendly language
    - Show empathy for patient concerns
    - Never provide medical advice or diagnosis
    """


def wrap_model(model: BaseChatModel) -> RunnableSerializable[PatientConsultationState, AIMessage]:
    bound_model = model.bind_tools(tools)
    preprocessor = RunnableLambda(
        lambda state: [SystemMessage(content=create_patient_consultation_prompt())] + state["messages"],
        name="StateModifier",
    )
    return preprocessor | bound_model


# Initialize the database helper
medical_db = MedicalDatabase()


async def initialize_consultation(state: PatientConsultationState, config: RunnableConfig, store: BaseStore) -> PatientConsultationState:
    """Initialize consultation state and load from SQLite database."""
    
    # Generate session ID if not present
    session_id = state.get("session_id") or str(uuid4())
    
    user_id = config["configurable"].get("user_id", "default")
    thread_id = config["configurable"].get("thread_id", "default")
    
    # Get or create consultation in SQLite database
    consultation_data = medical_db.get_or_create_consultation(session_id, user_id)
    
    # Convert stage string back to enum
    stage = ConsultationStage(consultation_data.get("consultation_stage", "greeting"))
    
    # Parse symptoms from JSON
    symptoms_json = consultation_data.get("symptoms_reported", "[]")
    try:
        symptoms = json.loads(symptoms_json) if symptoms_json else []
    except json.JSONDecodeError:
        symptoms = []
    
    patient_data = {
        "name": consultation_data.get("patient_name"),
        "email": consultation_data.get("patient_email"),
        "symptoms": symptoms,
        "responses": {},
        "stage": stage.value,
        "consultation_complete": consultation_data.get("completed", False)
    }
    
    return {
        "consultation_stage": stage,
        "patient_data": patient_data,
        "current_symptom_index": consultation_data.get("current_symptom_index", 0),
        "current_question_index": consultation_data.get("current_question_index", 0),
        "symptom_questions": [],
        "consultation_complete": consultation_data.get("completed", False),
        "session_id": session_id,
        "messages": []
    }


async def collect_basic_info(state: PatientConsultationState, config: RunnableConfig, store: BaseStore) -> PatientConsultationState:
    """Handle basic information collection and save to SQLite database."""
    
    last_message = state["messages"][-1] if state["messages"] else None
    user_input = last_message.content if isinstance(last_message, HumanMessage) else ""
    
    stage = state.get("consultation_stage", ConsultationStage.GREETING)
    patient_data = state.get("patient_data", {})
    session_id = state.get("session_id", str(uuid4()))
    
    if stage == ConsultationStage.GREETING:
        response = "Hello! I'm PatientBot, your medical consultation assistant. I'll help collect information about your symptoms for your healthcare provider.\n\nThis will take about 10-15 minutes. Let's start with your full name."
        new_stage = ConsultationStage.COLLECTING_BASIC_INFO
        
    elif stage == ConsultationStage.COLLECTING_BASIC_INFO:
        if not patient_data.get("name") and user_input.strip():
            patient_data["name"] = user_input.strip()
            response = f"Thank you, {patient_data['name']}. Now I need your email address for sending you the consultation summary."
            new_stage = stage
            
            # Save to database
            medical_db.update_consultation(
                session_id, 
                patient_name=patient_data["name"],
                consultation_stage=new_stage.value
            )
            
        elif not patient_data.get("email") and user_input.strip():
            # Basic email validation
            email = user_input.strip()
            if re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
                patient_data["email"] = email
                response = f"Perfect! Now, {patient_data['name']}, please tell me what symptoms or health concerns brought you here today. Describe them in your own words."
                new_stage = ConsultationStage.COLLECTING_SYMPTOMS
                
                # Save to database
                medical_db.update_consultation(
                    session_id,
                    patient_email=patient_data["email"],
                    consultation_stage=new_stage.value
                )
            else:
                response = "Please provide a valid email address (example: name@email.com)."
                new_stage = stage
        else:
            response = "Please provide the requested information so we can continue."
            new_stage = stage
            
    elif stage == ConsultationStage.COLLECTING_SYMPTOMS:
        if user_input.strip() and not patient_data.get("symptoms"):
            # Store symptoms
            symptoms = [s.strip() for s in user_input.split(",") if s.strip()]
            if not symptoms:
                symptoms = [user_input.strip()]
            patient_data["symptoms"] = symptoms
            
            response = f"I understand you're experiencing: {', '.join(symptoms)}.\n\nNow I'll ask you some specific questions about each symptom to help your healthcare provider better understand your condition."
            new_stage = ConsultationStage.ASKING_FOLLOWUP_QUESTIONS
            
            # Save to database
            medical_db.update_consultation(
                session_id,
                symptoms_reported=json.dumps(symptoms),
                consultation_stage=new_stage.value
            )
        else:
            response = "Please describe your symptoms or health concerns."
            new_stage = stage
    else:
        response = "Let me gather more information about your symptoms."
        new_stage = ConsultationStage.ASKING_FOLLOWUP_QUESTIONS
    
    consultation_complete = (new_stage == ConsultationStage.COMPLETED)
    
    return {
        "messages": [AIMessage(content=response)],
        "consultation_stage": new_stage,
        "patient_data": patient_data,
        "consultation_complete": consultation_complete,
        "session_id": session_id
    }


async def ask_followup_questions(state: PatientConsultationState, config: RunnableConfig) -> PatientConsultationState:
    """Ask follow-up questions using database search and save responses to SQLite."""
    
    patient_data = state.get("patient_data", {})
    symptoms = patient_data.get("symptoms", [])
    current_symptom_index = state.get("current_symptom_index", 0)
    current_question_index = state.get("current_question_index", 0)
    session_id = state.get("session_id", str(uuid4()))
    
    # If no symptoms, go to summary
    if not symptoms:
        return await generate_consultation_summary(state, config)
    
    # If we've asked enough questions (limit to 5 questions total), go to summary
    if current_question_index >= 5:
        return await generate_consultation_summary(state, config)
    
    # Get current symptom
    if current_symptom_index < len(symptoms):
        current_symptom = symptoms[current_symptom_index]
        
        # Simple predefined questions for each symptom
        standard_questions = [
            f"When did your {current_symptom} first start?",
            f"How would you describe your {current_symptom} (mild, moderate, severe)?",
            f"Does your {current_symptom} get worse with activity or at rest?",
            f"Have you taken any medication for your {current_symptom}?",
            "Do you have any other symptoms that occur along with this one?"
        ]
        
        if current_question_index < len(standard_questions):
            question = standard_questions[current_question_index]
            
            # Save the question to database for tracking
            question_id = f"q_{current_symptom_index}_{current_question_index}"
            
            # Update database
            medical_db.update_consultation(
                session_id,
                current_question_index=current_question_index + 1
            )
            
            return {
                "messages": [AIMessage(content=question)],
                "consultation_stage": ConsultationStage.ASKING_FOLLOWUP_QUESTIONS,
                "patient_data": patient_data,
                "current_question_index": current_question_index + 1,
                "consultation_complete": False,
                "session_id": session_id
            }
    
    # If we're done with questions, go to summary
    return await generate_consultation_summary(state, config)


async def store_response(state: PatientConsultationState, config: RunnableConfig) -> PatientConsultationState:
    """Store patient response to the previous question in SQLite database."""
    
    last_message = state["messages"][-1] if state["messages"] else None
    if isinstance(last_message, HumanMessage):
        user_input = last_message.content
        
        patient_data = state.get("patient_data", {})
        current_question_index = state.get("current_question_index", 0)
        current_symptom_index = state.get("current_symptom_index", 0)
        session_id = state.get("session_id", str(uuid4()))
        symptoms = patient_data.get("symptoms", [])
        
        # Save the response to SQLite database
        if symptoms and current_symptom_index < len(symptoms):
            current_symptom = symptoms[current_symptom_index]
            question_id = f"q_{current_symptom_index}_{current_question_index - 1}"
            
            # Determine which question was asked
            standard_questions = [
                f"When did your {current_symptom} first start?",
                f"How would you describe your {current_symptom} (mild, moderate, severe)?",
                f"Does your {current_symptom} get worse with activity or at rest?",
                f"Have you taken any medication for your {current_symptom}?",
                "Do you have any other symptoms that occur along with this one?"
            ]
            
            question_idx = (current_question_index - 1) % len(standard_questions)
            if question_idx < len(standard_questions):
                question_text = standard_questions[question_idx]
                
                # Save response to database
                medical_db.save_patient_response(
                    session_id=session_id,
                    question_id=question_id,
                    question_text=question_text,
                    category="symptom_details",
                    response_text=user_input,
                    symptom_name=current_symptom
                )
        
        return {
            "messages": [],
            "patient_data": patient_data,
            "session_id": session_id
        }
    
    return {"messages": []}


async def generate_consultation_summary(state: PatientConsultationState, config: RunnableConfig) -> PatientConsultationState:
    """Generate the final consultation summary and save to SQLite database."""
    
    patient_data = state.get("patient_data", {})
    session_id = state.get("session_id", str(uuid4()))
    
    # Get all responses from database
    responses = medical_db.get_consultation_responses(session_id)
    
    summary = f"""
## Consultation Summary

**Patient Information:**
- Name: {patient_data.get('name', 'Not provided')}
- Email: {patient_data.get('email', 'Not provided')}
- Date: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}

**Reported Symptoms:**
{chr(10).join([f"• {symptom}" for symptom in patient_data.get('symptoms', [])])}

**Responses Collected:**
{chr(10).join([f"• {resp['question']}: {resp['response']}" for resp in responses])}

---

**Next Steps:**
1. Schedule an appointment with your healthcare provider
2. Bring this summary to your appointment
3. If you have emergency symptoms, seek immediate medical attention

**Important:** This consultation collected information only. Please consult healthcare professionals for proper diagnosis and treatment.

Thank you for your time, {patient_data.get('name', '')}! Your consultation is now complete.
    """
    
    # Save summary to database
    medical_db.save_consultation_summary(
        session_id=session_id,
        summary_text=summary.strip(),
        key_findings=[resp['response'] for resp in responses if 'severe' in resp['response'].lower()],
        recommendations=[
            "Schedule appointment with healthcare provider",
            "Bring consultation summary to appointment",
            "Seek immediate care for emergency symptoms"
        ],
        next_steps=[
            "Contact primary care physician",
            "Review summary before appointment",
            "Monitor symptoms"
        ]
    )
    
    # Mark consultation as completed
    medical_db.update_consultation(
        session_id,
        completed=True,
        consultation_end_time=datetime.now().isoformat(),
        consultation_stage=ConsultationStage.COMPLETED.value
    )
    
    return {
        "messages": [AIMessage(content=summary.strip())],
        "consultation_stage": ConsultationStage.COMPLETED,
        "patient_data": patient_data,
        "consultation_complete": True,
        "session_id": session_id
    }


# Define the graph
agent = StateGraph(PatientConsultationState)
agent.add_node("initialize", initialize_consultation)
agent.add_node("collect_basic_info", collect_basic_info)
agent.add_node("store_response", store_response)
agent.add_node("ask_followup", ask_followup_questions)

# Set entry point
agent.set_entry_point("initialize")

# Add edges
agent.add_edge("initialize", "collect_basic_info")

def route_after_basic_info(state: PatientConsultationState) -> Literal["collect_basic_info", "ask_followup", END]:
    """Route after collecting basic info."""
    stage = state.get("consultation_stage")
    
    if state.get("consultation_complete"):
        return END
    elif stage == ConsultationStage.ASKING_FOLLOWUP_QUESTIONS:
        return "ask_followup"
    else:
        return "collect_basic_info"

def route_after_followup(state: PatientConsultationState) -> Literal["store_response", END]:
    """Route after follow-up questions."""
    if state.get("consultation_complete"):
        return END
    else:
        return "store_response"

# Add conditional edges
agent.add_conditional_edges("collect_basic_info", route_after_basic_info, {
    "collect_basic_info": "collect_basic_info",
    "ask_followup": "ask_followup", 
    END: END
})

agent.add_conditional_edges("ask_followup", route_after_followup, {
    "store_response": "store_response",
    END: END
})

agent.add_edge("store_response", "ask_followup")

# Compile the agent
patient_consultation_agent = agent.compile()