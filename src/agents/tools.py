# import math
# import re

# import numexpr
# from langchain_chroma import Chroma
# from langchain_core.tools import BaseTool, tool
# from langchain_openai import OpenAIEmbeddings


# def calculator_func(expression: str) -> str:
#     """Calculates a math expression using numexpr.

#     Useful for when you need to answer questions about math using numexpr.
#     This tool is only for math questions and nothing else. Only input
#     math expressions.

#     Args:
#         expression (str): A valid numexpr formatted math expression.

#     Returns:
#         str: The result of the math expression.
#     """

#     try:
#         local_dict = {"pi": math.pi, "e": math.e}
#         output = str(
#             numexpr.evaluate(
#                 expression.strip(),
#                 global_dict={},  # restrict access to globals
#                 local_dict=local_dict,  # add common mathematical functions
#             )
#         )
#         return re.sub(r"^\[|\]$", "", output)
#     except Exception as e:
#         raise ValueError(
#             f'calculator("{expression}") raised error: {e}.'
#             " Please try again with a valid numerical expression"
#         )


# calculator: BaseTool = tool(calculator_func)
# calculator.name = "Calculator"


# # Format retrieved documents
# def format_contexts(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# def load_chroma_db():
#     # Create the embedding function for our project description database
#     try:
#         embeddings = OpenAIEmbeddings()
#     except Exception as e:
#         raise RuntimeError(
#             "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
#         ) from e

#     # Load the stored vector database
#     chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
#     retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
#     return retriever


# def database_search_func(query: str) -> str:
#     """Searches chroma_db for information in the company's handbook."""
#     # Get the chroma retriever
#     retriever = load_chroma_db()

#     # Search the database for relevant documents
#     documents = retriever.invoke(query)

#     # Format the documents into a string
#     context_str = format_contexts(documents)

#     return context_str


# database_search: BaseTool = tool(database_search_func)
# database_search.name = "Database_Search"  # Update name with the purpose of your database
import math
import re
import json
from typing import List, Dict, Any

import numexpr
from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langchain_openai import OpenAIEmbeddings


def calculator_func(expression: str) -> str:
    """Calculates a math expression using numexpr.

    Useful for when you need to answer questions about math using numexpr.
    This tool is only for math questions and nothing else. Only input
    math expressions.

    Args:
        expression (str): A valid numexpr formatted math expression.

    Returns:
        str: The result of the math expression.
    """

    try:
        local_dict = {"pi": math.pi, "e": math.e}
        output = str(
            numexpr.evaluate(
                expression.strip(),
                global_dict={},  # restrict access to globals
                local_dict=local_dict,  # add common mathematical functions
            )
        )
        return re.sub(r"^\[|\]$", "", output)
    except Exception as e:
        raise ValueError(
            f'calculator("{expression}") raised error: {e}.'
            " Please try again with a valid numerical expression"
        )


calculator: BaseTool = tool(calculator_func)
calculator.name = "Calculator"


# Format retrieved documents
def format_contexts(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_chroma_db():
    # Create the embedding function for our project description database
    try:
        embeddings = OpenAIEmbeddings()
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize OpenAIEmbeddings. Ensure the OpenAI API key is set."
        ) from e

    # Load the stored vector database
    chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 5})
    return retriever


def database_search_func(query: str) -> str:
    """Searches chroma_db for information in the company's handbook."""
    # Get the chroma retriever
    retriever = load_chroma_db()

    # Search the database for relevant documents
    documents = retriever.invoke(query)

    # Format the documents into a string
    context_str = format_contexts(documents)

    return context_str


def get_symptom_followup_questions_func(symptom: str) -> str:
    """
    Retrieves structured follow-up questions for a specific symptom from the medical database.
    
    Args:
        symptom (str): The symptom to search for (e.g., "chest pain", "shortness of breath")
    
    Returns:
        str: JSON string containing organized follow-up questions by category
    """
    try:
        # Get the chroma retriever
        retriever = load_chroma_db()
        
        # Search specifically for medical symptoms
        query = f"medical symptom {symptom} follow-up questions assessment"
        documents = retriever.invoke(query)
        
        # Filter for medical symptom documents
        medical_docs = [
            doc for doc in documents 
            if doc.metadata.get("type") == "medical_symptom" 
            and symptom.lower() in doc.metadata.get("symptom", "").lower()
        ]
        
        if not medical_docs:
            # Fallback to general search
            medical_docs = [
                doc for doc in documents 
                if "symptom" in doc.page_content.lower() 
                and symptom.lower() in doc.page_content.lower()
            ]
        
        if medical_docs:
            # Extract the most relevant document
            best_doc = medical_docs[0]
            
            # Parse the content to extract structured questions
            questions_data = parse_symptom_questions(best_doc.page_content, symptom)
            
            return json.dumps(questions_data, indent=2)
        else:
            return json.dumps({
                "symptom": symptom,
                "questions": [],
                "message": f"No specific follow-up questions found for {symptom}. Please use general assessment questions."
            })
            
    except Exception as e:
        return json.dumps({
            "error": f"Error retrieving questions for {symptom}: {str(e)}",
            "questions": []
        })


def parse_symptom_questions(content: str, symptom: str) -> Dict[str, Any]:
    """
    Parse the medical symptom content to extract structured questions.
    
    Args:
        content (str): The raw content from the database
        symptom (str): The symptom being queried
    
    Returns:
        Dict: Structured questions organized by category
    """
    questions_by_category = {}
    current_category = None
    
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Identify category headers
        if line.endswith(':') and any(keyword in line.lower() for keyword in 
                                    ['symptom details', 'vital signs', 'medical history', 
                                     'red flags', 'lifestyle', 'psychosocial']):
            current_category = line.rstrip(':')
            questions_by_category[current_category] = []
        
        # Extract questions (lines starting with numbers or bullet points)
        elif current_category and (line.startswith(('1.', '2.', '3.', '4.', '5.', '-', '•')) or 
                                 line.endswith('?')):
            # Clean up the question text
            question = re.sub(r'^\d+\.\s*', '', line)
            question = re.sub(r'^[-•]\s*', '', question)
            question = question.strip()
            
            if question and len(question) > 10:  # Filter out very short lines
                questions_by_category[current_category].append({
                    "text": question,
                    "category": current_category,
                    "priority": get_question_priority(current_category)
                })
    
    # Structure the response
    return {
        "symptom": symptom,
        "total_questions": sum(len(questions) for questions in questions_by_category.values()),
        "categories": list(questions_by_category.keys()),
        "questions_by_category": questions_by_category,
        "prioritized_questions": get_prioritized_questions(questions_by_category)
    }


def get_question_priority(category: str) -> int:
    """Assign priority to question categories (1 = highest priority)."""
    priority_map = {
        "Red Flags Questions": 1,
        "Vital Signs Questions": 2,
        "Symptom Details Questions": 3,
        "Medical History Questions": 4,
        "Past Medical History Questions": 5,
        "Lifestyle Risk Factors Questions": 6,
        "Psychosocial Questions": 7
    }
    
    # Find matching category (case-insensitive, partial match)
    for key, priority in priority_map.items():
        if any(word in category.lower() for word in key.lower().split()):
            return priority
    
    return 8  # Default priority


def get_prioritized_questions(questions_by_category: Dict[str, List[Dict]]) -> List[Dict]:
    """Get all questions sorted by priority."""
    all_questions = []
    
    for category, questions in questions_by_category.items():
        for question in questions:
            all_questions.append(question)
    
    # Sort by priority, then by category
    all_questions.sort(key=lambda x: (x.get("priority", 8), x.get("category", "")))
    
    return all_questions


def get_patient_friendly_questions_func(symptom: str, max_questions: int = 10) -> str:
    """
    Get patient-friendly follow-up questions for a specific symptom.
    
    Args:
        symptom (str): The symptom to get questions for
        max_questions (int): Maximum number of questions to return
    
    Returns:
        str: Formatted string of patient-friendly questions
    """
    try:
        # Get structured questions
        questions_json = get_symptom_followup_questions_func(symptom)
        questions_data = json.loads(questions_json)
        
        if "error" in questions_data:
            return f"I couldn't find specific questions for {symptom}. Let me ask you some general questions about your symptoms."
        
        # Get prioritized questions
        prioritized_questions = questions_data.get("prioritized_questions", [])
        
        if not prioritized_questions:
            return f"Let me ask you some questions about your {symptom}."
        
        # Format for patient interaction
        selected_questions = prioritized_questions[:max_questions]
        
        formatted_questions = []
        for i, q in enumerate(selected_questions, 1):
            # Make questions more patient-friendly
            question_text = make_patient_friendly(q["text"])
            formatted_questions.append(f"{i}. {question_text}")
        
        return f"I'd like to ask you some questions about your {symptom}:\n\n" + "\n".join(formatted_questions)
        
    except Exception as e:
        return f"Let me ask you about your {symptom}. When did it start?"


def make_patient_friendly(question: str) -> str:
    """Convert medical questions to patient-friendly language."""
    
    # Replace medical terms with simpler language
    replacements = {
        "orthopnea": "difficulty breathing when lying down",
        "paroxysmal nocturnal dyspnea": "suddenly waking up gasping for air",
        "dyspnea": "shortness of breath",
        "palpitations": "heart racing or irregular heartbeat",
        "syncope": "fainting or losing consciousness",
        "diaphoresis": "sweating",
        "hemoptysis": "coughing up blood",
        "oliguria": "decreased urination",
        "nocturia": "frequent nighttime urination",
        "bradycardia": "slow heart rate",
        "tachycardia": "fast heart rate",
    }
    
    question_lower = question.lower()
    for medical_term, friendly_term in replacements.items():
        question_lower = question_lower.replace(medical_term, friendly_term)
    
    # Capitalize first letter
    if question_lower:
        question = question_lower[0].upper() + question_lower[1:]
    
    return question


# Create the tool instances
database_search: BaseTool = tool(database_search_func)
database_search.name = "Database_Search"

get_symptom_followup_questions: BaseTool = tool(get_symptom_followup_questions_func)
get_symptom_followup_questions.name = "Get_Symptom_Followup_Questions"

get_patient_friendly_questions: BaseTool = tool(get_patient_friendly_questions_func)
get_patient_friendly_questions.name = "Get_Patient_Friendly_Questions"