import utils
import json
import random
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from utils import db


guess_species_router = APIRouter()

class AnswerSubmission(BaseModel):
    sid:int
    answer: str
    hint_used: bool


class QUIZ(BaseModel):
    """Input Species data into a quiz output"""
    question: str = Field(..., description="The Question to be asked for quiz")
    hint: str = Field(..., description="The hint for the question in one line")
    answer: str = Field(..., description="The correct asnwer for the quiz")

# llm = ChatOllama(
#     model=utils.MODEL_LLAMA,
#     temperature=0.0,
#     disable_streaming=False,
#     seed=43,
#     num_predict=100
# )

# llm_with_structure = llm.with_structured_output(QUIZ)

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
        f"the data is {q_data["text"]}. answer is {q_data["species_name"]} belongs to {q_data["genus"]} genus"
    ))
    #response:QUIZ = llm_with_structure.invoke(conversation)
    response = {"question":"test","hint":'dsds',"answer":"ffd dd 123" }
    print(response)
    # return {
    #     "question": response.question,  #NOTE change
    #     "hint": response.hint,
    #     "ans" : response.answer
    # }
    return {
        "question": response["question"],  #NOTE change
        "hint": response["hint"],
        "ans" : response["answer"]
    }

@guess_species_router.post("/checkAnswer")
def check_answer(submission: AnswerSubmission):
    global current_question_index
    
    target_data = questions_db[current_question_index]
    actual_species_name = target_data.get("species_name") 

    if submission.sid <= 0:
        raise HTTPException(status_code=401,detail="Invalid user Id")

    if submission.answer == "correct":
        awarded_points = 2 if submission.hint_used else 5
        
        # --- DATABASE UPDATE LOGIC ---
        # You can now use submission.sid to update the specific user
        print(f"Updating points for Student ID: {submission.sid}")
        points = db.fetchone("SELECT score from Leaderboard where sid=?",submission.sid)
        value = points[0] if points else 0
        if not value:
            db.execute("INSERT INTO leaderboard VALUES(?,?)", submission.sid, awarded_points)
        else:
            db.execute("UPDATE leaderboard SET score=? where sid=?", value+awarded_points, submission.sid)

        # update_student_score(student_id=submission.sid, points=awarded_points)
        # -----------------------------
        return {
            "result": True, 
            "points": awarded_points
        }
    else:
        return {
            "result": False, 
            "points": 0, 
            "correct_answer": actual_species_name
        }