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
    hint: int = Field(..., description="The hint for the question in one line")
    answer: str = Field(..., description="The correct asnwer for the quiz")

llm = ChatOllama(
    model=utils.MODEL_LLAMA,
    temperature=0.0,
    disable_streaming=False,
    seed=43,
    num_predict=100
)

llm_with_structure = llm.with_structured_output(QUIZ)


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
    return {
        "question": q_data["question"], 
        "hint": q_data["hint"]
    }

@guess_species_router.post("/checkAnswer")
def check_answer(submission: AnswerSubmission):
    global current_question_index
    
    target_data = questions_db[current_question_index]
    correct_answer = target_data["a"].lower().strip()
    user_answer = submission.answer.lower().strip()
    
    if user_answer == correct_answer:
        awarded_points = 2 if submission.hint_used else 5
        
        return {
            "result": True, 
            "points": awarded_points
        }
    else:
        return {
            "result": False, 
            "points": 0
        }