from fastapi import APIRouter
from pydantic import BaseModel, EmailStr


student_router = APIRouter(prefix = "/student", tags=["Student"])

class LOGIN(BaseModel):
    username: str
    password: str

class REGISTER(BaseModel):
    username: str
    email: EmailStr
    institution: str
    password: str



@student_router.post("/login")
async def student_login(data: LOGIN):
    # Perform your DB check here
    # If success:
    return {
        "message": "Login successful", 
        "uid":99, 
        "user": data.username, 
        "type" : "student"
    }
    # If fail:
    # raise HTTPException(status_code=401, detail="Invalid credentials")

@student_router.post("/register")
async def student_register(data: REGISTER):
    # Perform your DB insertion here
    # If success:
    return {
        "message": "User registered successfully", 
        "uid":99, 
        "user": data.username, 
        "type" : "student"
    }
    # If exists:
    # raise HTTPException(status_code=400, detail="User already exists")



