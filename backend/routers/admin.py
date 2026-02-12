from fastapi import FastAPI, APIRouter
from pydantic import BaseModel, EmailStr


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
    # Perform your DB check here
    # If success:
    return {
        "message": "Login successful", 
        "uid":99, 
        "user": data.username, 
        "type" : "admin"
    }
    # If fail:
    # raise HTTPException(status_code=401, detail="Invalid credentials")
