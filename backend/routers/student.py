from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from utils import db
import sqlite3

student_router = APIRouter(prefix = "/student", tags=["Student"])

class LOGIN(BaseModel):
    email: EmailStr
    password: str

class REGISTER(BaseModel):
    username: str
    email: EmailStr
    institution: str
    password: str



@student_router.post("/login")
async def student_login(data: LOGIN):
    print(data)

    user = db.fetchone(
        "SELECT sid, username, password FROM student WHERE email = ?",
        data.email
    )

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    sid, username, password = user  # (1, 'Archana', '123')

    if password != data.password:
        raise HTTPException(status_code=401, detail="Invalid password")


    return {
        "message": "Login successful", 
        "uid":sid, 
        "user": username, 
        "type" : "student"
    }
 

@student_router.post("/register")
async def student_register(data: REGISTER):

    try:
        sid = db.execute(
            """INSERT INTO student(username, email, institution, password) VALUES(?,?,?,?)""",
            data.username, data.email, data.institution, data.password 
        )

        return {
            "message": "User registered successfully", 
            "uid": sid, 
            "user": data.username, 
            "type" : "student"
        }

    except sqlite3.IntegrityError as e:
        raise HTTPException(
            status_code=400, 
            detail="Email already exists" if "email" in str(e) else str(e)
        )




