from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr
from fastapi.security import OAuth2PasswordRequestForm
from ml.serving.auth import get_password_hash, verify_password, create_access_token, get_current_user
from ml.serving.database import get_db

router = APIRouter(prefix="/auth", tags=["auth"])

class UserRegister(BaseModel):
    name: str
    email: EmailStr
    password: str

@router.post("/register")
async def register(user: UserRegister, db = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    existing_user = await db["users"].find_one({"email": user.email})
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    import uuid
    from datetime import datetime
    new_user = {
        "_id": uuid.uuid4().hex,
        "name": user.name,
        "email": user.email,
        "password_hash": get_password_hash(user.password),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    await db["users"].insert_one(new_user)
    return {"message": "User registered successfully"}

@router.post("/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db = Depends(get_db)):
    if db is None:
        raise HTTPException(status_code=503, detail="Database not available")
    user = await db["users"].find_one({"email": form_data.username})
    if not user or not verify_password(form_data.password, user["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": user["_id"]})
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/logout")
async def logout():
    return {"message": "Logged out successfully"}

@router.get("/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user
