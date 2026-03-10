from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from utils import db


admin_router = APIRouter(prefix = "/admin", tags=["Admin"])


class LOGIN(BaseModel):
    username: str
    password: str

# class REGISTER(BaseModel):
#     username: str
#     email: EmailStr
#     institution: str
#     password: str

@admin_router.post("/login")
async def admin_login(data: LOGIN):
    print(data)

    user = db.fetchone(
        "SELECT aid, username, password FROM admin WHERE username = ?",
        data.username
    )

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    aid, username, password = user  # (1, 'Archana', '123')

    if password != data.password:
        raise HTTPException(status_code=401, detail="Invalid password")


    return {
        "message": "Login successful", 
        "uid":aid, 
        "user": username, 
        "type" : "admin"
    }
