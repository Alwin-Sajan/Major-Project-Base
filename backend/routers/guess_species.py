import utils
import json
import random
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from fastapi import APIRouter
from pydantic import BaseModel, Field


guess_species_router = APIRouter()

class AnswerSubmission(BaseModel):
    answer: str
    hint_used: bool


class QUIZ(BaseModel):
    """Input Species data into a quiz output"""
    question: str = Field(..., description="The Question to be asked for quiz")
    hint: str = Field(..., description="The hint for the question in one line")
    answer: str = Field(..., description="The correct asnwer for the quiz")

llm = ChatOllama(
    model=utils.MODEL_LLAMA,
    temperature=0.0,
    disable_streaming=False,
    seed=43,
    num_predict=100
)

llm_with_structure = llm.with_structured_output(QUIZ)

conversation = [
    SystemMessage(
        "You are a Quiz expert, your task is create a question from a given data, also generate hint for the question,"
        "answer will be given along with the content. content contains species name and some text on its features"
        "Create a question to identify species(state all features) and a suitable hint. Must be easy to predict, hints can be more revealing to help students"
    )
]


def load_taxonomic_for_game(jsonl_path: str = utils.JSONL_RAG_PATH):
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("chunk_type") != "taxonomic":
                continue
            entries.append({
                "species_name": data.get("species_name"),
                "genus": data.get("genus"),
                "text": data.get("text"),
            })
    return entries

questions_db = load_taxonomic_for_game()

current_question_index = 0

@guess_species_router.get("/getQuestion")
def get_question():
    global current_question_index

    current_question_index = random.randint(0, len(questions_db) - 1)
    q_data = questions_db[current_question_index]

    conversation.append(HumanMessage(
        f"the data is {q_data["text"]}. answer is {q_data["text"]} belongs to {q_data["genus"]} genus"
    ))
    response:QUIZ = llm_with_structure.invoke(conversation)
    print(response)
    return {
        "question": response.question, 
        "hint": response.hint,
        "ans" : response.answer
    }

@guess_species_router.post("/checkAnswer")
def check_answer(submission: AnswerSubmission):
    global current_question_index
    
    # Get the real answer from your DB to show the student if they are wrong
    target_data = questions_db[current_question_index]
    actual_species_name = target_data["a"] 

    if submission.answer == "correct":
        # Calculate points
        awarded_points = 2 if submission.hint_used else 5
        
        # --- DATABASE UPDATE LOGIC HERE ---
        # update_user_points(user_id, awarded_points)
        # ----------------------------------
        
        return {
            "result": True, 
            "points": awarded_points
        }
    else:
        # User was incorrect, return the learning data
        return {
            "result": False, 
            "points": 0, 
            "correct_answer": actual_species_name
        }