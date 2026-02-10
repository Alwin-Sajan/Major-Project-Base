from fastapi import APIRouter
from pydantic import BaseModel
import random

guess_species_router = APIRouter()

class AnswerSubmission(BaseModel):
    answer: str
    hint_used: bool


questions_db = [
    {"q": "Which marine species is known as the 'sea canary'?", "h": "It is a white whale found in the Arctic.", "a": "beluga"},
    #{"q": "What is the largest animal to ever live on Earth?", "h": "It's a marine mammal that eats krill.", "a": "blue whale"},
    #{"q": "Which fish is famous for its vertical swimming and prehensile tail?", "h": "The males carry the eggs in a pouch.", "a": "seahorse"}
]

current_question_index = 0

@guess_species_router.get("/getQuestion")
def get_question():
    global current_question_index

    current_question_index = random.randint(0, len(questions_db) - 1)
    q_data = questions_db[current_question_index]
    return {"question": q_data["q"], "hint": q_data["h"]}

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