from fastapi import FastAPI, Request, Depends, UploadFile, BackgroundTasks, status
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic.networks import EmailStr
import uvicorn
from sqlalchemy import func
import os
import models
from database import engine, SessionLocal
from sqlalchemy.orm import Session
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from models import Confession, User, Comment
from schema import ConfessionCreate, ConfessionResponse, UserCreate, UserResponse, CommentCreate, CommentResponse, ForgotPasswordRequest, ResetPasswordRequest, MarkAsReadRequest
from sqlalchemy.exc import IntegrityError

from typing import List, Optional, Dict, Any, Tuple
import re
from dotenv import load_dotenv
import os
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi.responses import StreamingResponse
import asyncio
import json
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from bisect import bisect_left
import cloudinary
import cloudinary.uploader
from cloudinary.utils import cloudinary_url
import google.generativeai as genai
import logging
from functools import lru_cache

from data import emails_list, name_list
from utils import analyze_confession_with_llm

#-----------------------------------------Do Not Change here----------------------------------------------------------------------
load_dotenv() 
models.Base.metadata.create_all(bind=engine)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#SECRETS that has not to be shared on the github and has to stored in .env file only
SECRET_KEY = os.getenv(key="SECRET_KEY")
ALGORITHM = os.getenv(key="ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTE = int(os.getenv(key="ACCESS_TOKEN_EXPIRE_MINUTE"))
MAIL = os.getenv(key="MAIL")
MAIL_PASSWORD = os.getenv(key="MAIL_PASSWORD")
SALT = os.getenv(key="PASSWORD_RESET_SALT")
CLOUD_NAME = os.getenv(key="CLOUD_NAME")
API_KEY_CLOUD = os.getenv(key="API_KEY_CLOUD")
API_SECRET = os.getenv(key="API_SECRET")
GEMINI_API_KEY = os.getenv(key="GEMINI_API_KEY")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configure Google Gemini API
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    logger.warning("GEMINI_API_KEY not found in environment variables. Confession moderation will fall back to default behavior.")



conf = ConnectionConfig(
    MAIL_USERNAME=MAIL, 
    MAIL_PASSWORD=MAIL_PASSWORD,
    MAIL_FROM=MAIL,
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,  # Use either MAIL_SSL_TLS or MAIL_STARTTLS, not both
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True
)
cloudinary.config( 
    cloud_name = CLOUD_NAME, 
    api_key = API_KEY_CLOUD, 
    api_secret = API_SECRET,
    secure=True
)

# Serializer for token generation
serializer = URLSafeTimedSerializer(SECRET_KEY)

images_path = Path("images")
app.mount("/images", StaticFiles(directory=images_path), name="images")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")
pwd_context = CryptContext(schemes=['bcrypt'], deprecated="auto")

comment_event_queue = asyncio.Queue()

#dependancey
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------------------------------------------------------------------------------------------------------------------------

  
# -------------Functions for authentication---------------------------------------------------------------------------------------------------
def get_user_by_username(username: str, db: Session):
    return db.query(
        models.User).filter(models.User.username == username).first()


def get_user_by_email(email: str, db: Session):
    return db.query(models.User).filter(models.User.email == email).first()


def create_user(user: UserCreate, db: Session):
    hashedpassword = pwd_context.hash(
        user.password)  # Plain password from the frontend
    try:
        user_model = db.query(
            models.User).filter(models.User.username == user.username).first()
        if user_model is not None:
            return JSONResponse(status_code=400,
                                content={"message": f"Username already exists"})

        user_model = db.query(
            models.User).filter(models.User.email == user.email).first()
        if user_model is not None:
            return JSONResponse(status_code=400,
                                content={"message": f"Email already exists"})
        global emails_list, name_list
        index = bisect_left(list(emails_list), user.email)

        user_model = models.User(username=user.username,
                                 email=user.email,
                                 hashedpassword=hashedpassword,
                                 name = name_list[index]
        )
        db.add(user_model)
        db.commit()
        return JSONResponse(
            status_code=200,
            content={
                "message":
                f"User created successfully with username: {user.username}"
            })
    except Exception as e:
        print("Error: ", e)
        raise HTTPException(status_code=500, detail=f"Something went wrong")


def authenticate_user(email: EmailStr, password: str, db: Session):
    user = db.query(User).filter(User.email == str(email)).first()
    if not user:
        print("email is wrong")
        return None
    verified_user = pwd_context.verify(password, str(user.hashedpassword))
    if verified_user: 
        return user
    return None
    


def create_access_token(data: dict, expire_delta: timedelta | None = None):
    to_encode = data.copy()
    if expire_delta:
        expire = datetime.now(timezone.utc) + expire_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=24 * 30 * 60 * 60)  # 30 days
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(payload=to_encode,
                             key=SECRET_KEY,
                             algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str, db:Session):
    try:
        payload = jwt.decode(jwt=token, key=SECRET_KEY, algorithms=["HS256"])
        id: int = payload.get("id")
        user = db.query(User).filter(models.User.id == id).first()
        if not user:
            raise HTTPException(status_code=403, detail=f"User not found")
        return payload
    except Exception as e:
        print("Exception occured: ", e)
        raise HTTPException(status_code=401,
                            detail=f"Token is invalid or expired")


async def get_current_user(token: str = Depends(oauth2_scheme),
                           db: Session = Depends(get_db)):
    try:
        payload = jwt.decode(jwt=token, key=SECRET_KEY, algorithms=["HS256"])
        id: int = payload.get("id")
        user = db.query(models.User).filter(models.User.id == id).first()
        return user
    except Exception as e:
        print("Exception occured: ", e)
        raise HTTPException(status_code=401,
                            detail=f"Token is invalid or expired")


# ------------------------------------------------------------------------------------------------------------------------


# ----------------------------------------------Auth Routes---------------------------------
@app.post("/signup")
async def register_user(user: UserCreate, db: Session = Depends(get_db)):
    user_model = get_user_by_username(user.username, db)
    if user_model is not None:
        return JSONResponse(status_code=400,
                            content={"message": "Username already taken"})
    user_model = get_user_by_email(user.email, db)
    if user_model:
        return JSONResponse(status_code=400,
                            content={"message": f"Email already exists"})
    if user.email not in emails_list:
        return JSONResponse(status_code=400,
                            content={"message": f"This email doesn't exists in our database please enter your college mail"})
    return create_user(user, db)


#------------------Login Route-----------------------
# email has to be there
@app.post("/login")
async def login_for_accesstoken(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: Session = Depends(get_db)):
    user = authenticate_user(email=form_data.username, # email is considered username here
                             password=form_data.password,
                             db=db)
    if user is None:
        raise HTTPException(status_code=401, detail=f"Incorrect password or email")
    access_token_expire = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTE)
    access_token = create_access_token(data={"id": user.id},
                                       expire_delta=access_token_expire)
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/verifytoken")
async def verify_user_token(token: str, db:Session=Depends(get_db)):
    verify_token(token, db)
    return {"message": "Token is Valid"}

#------------------------Forgot Password Routes-------------------------

@app.post("/forgot-password")
async def forgot_password(request: ForgotPasswordRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.email == request.email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User with this email does not exist."
        )

    # Generate a token that expires in 10 minutes
    token = serializer.dumps(user.email, salt=SALT)
    reset_link = f"https://avowal-backend.vercel.app/reset-password?token={token}"

    # Prepare the email
    message = MessageSchema(
        subject="Password Reset Request",
        recipients=[request.email],
        body=f"Click on the link to reset your password: {reset_link}",
        subtype="html"
    )

    # Send email
    fm = FastMail(conf)
    background_tasks.add_task(fm.send_message, message)

    return {"message": "Password reset email has been sent."}

@app.post("/reset-password")
async def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    try:
        # Decode the token (expires in 10 minutes)
        email = serializer.loads(request.token, salt=SALT, max_age=600)
    except SignatureExpired:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The token has expired."
        )
    except BadTimeSignature:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid token."
        )

    # Fetch user and update the password
    user = db.query(models.User).filter(models.User.email == email).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found."
        )

    # Update the password (hash the password as per your application logic)
    print(request.new_password)
    user.hashedpassword = pwd_context.hash(request.new_password)# Make sure to hash this password before storing
    db.commit()
    return {"message": "Password has been reset successfully."}

#-------------------------Routes for user profile--------------------------------------

# User can only update username and relationship_status
@app.put("/update") 
async def update_user(username: Optional[str]=None, relationship_status:Optional[str]=None,
                      db: Session = Depends(get_db),
                      current_user: models.User = Depends(get_current_user)):
    if relationship_status:
        if relationship_status not in ["Single", "Committed"]:
            raise HTTPException(status_code=400, detail=f"Relationship status can be either Single or Committed")
        current_user.relationship_status = relationship_status
    if username:
        user = get_user_by_username(username, db)
        if user and user!=current_user:
            raise HTTPException(status_code=403, detail=f"Username already taken")
        db.query(User).filter(User.id == current_user.id).update({"username": username})
    if relationship_status:
        db.query(User).filter(User.id == current_user.id).update({"relationship_status": relationship_status})
    if username or relationship_status:
        db.commit()
        db.refresh(current_user)
    user = current_user
    data = jsonable_encoder(user, include=["id", "username", "relationship_status", "name"])
    return {"message": "User updated successfully", "data": data}


# User can only update profile pic
@app.post("/update_profile_pic")
async def upload_profile_pic(
    file: UploadFile,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):

    if file.headers.get('content-type').find("image") == -1:
        return JSONResponse(
            status_code=404,
            content={"message": "File provided is not an image"})
    
    # delete previous file from cloud
    if current_user.profile_pic != "images/profile/def.jpg":
        cloudinary.uploader.destroy(current_user.username)
        
    filecontent = await file.read()
    
    ext = file.filename.split(".")[-1]
    if ext not in ["jpg", "jpeg", "png"]:       
        return JSONResponse(status_code=404,
            content={"message": "File provided is not an image"})
        
    # filename = "pic_" + str(current_user.id) + "." + ext
    # os.makedirs("images/profile", exist_ok=True) 
    # path = os.path.join(os.curdir, "profile")
    # try:
    #     os.makedirs(path)
    # except FileExistsError:
    #     print("File already exists")
    # filename = f"images/profile/{filename}"
    # with open(filename, mode="wb") as f:
    #     f.write(filecontent)
    
    upload_result = cloudinary.uploader.upload(filecontent,public_id=current_user.username, eager=[{"width": 500, "height": 500, "crop": "thumb", 'gravity': "auto", "aspect_ratio": "1.0", 'radius': 10}])
    db.query(models.User).filter(models.User.id == current_user.id).update({"profile_pic": upload_result["eager"][0]["secure_url"]})
    db.commit()
    return JSONResponse(status_code=200,
        content={
            "message": "Profile pic updated",
            "data": {
                "url": upload_result["eager"][0]["secure_url"]
            }
        })


@app.get("/profile_data")
async def get_profile(current_user: models.User = Depends(get_current_user)):
    data = jsonable_encoder(current_user, exclude=["hashedpassword", "id", "unread_confessions"])
    return {"message": "Profile fetched successfully", "data": data}


@app.get("/search_users")
async def search_users(q: str, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    users = db.query(models.User).filter(
        models.User.username.like(f"{q}%")).all()
    s = set(users)
    
    user_list = db.query(models.User).filter(
        models.User.email.like(f"{q}%")).all()
    for user in user_list:
        s.add(user)
        
    data = [
        jsonable_encoder(user, exclude=["hashedpassword", "id", "unread_confessions"])
        for user in s
    ]
    return {"message": "Users found", "data": data}


@app.get("/user")  # viewed profile function yet to be implemented
async def get_user(username: str, db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    user = get_user_by_username(username, db)
    if not user:
        raise HTTPException(status_code=403, detail=f"Username not found")
    data = jsonable_encoder(user, exclude=["hashedpassword", "id", "unread_confessions"])
    
    return {"message": "User found", "data": data}


@app.delete("/delete_user")
async def delete_user(db: Session = Depends(get_db), current_user: models.User = Depends(get_current_user)):
    db.query(models.User).filter(
        models.User.id == current_user.id).delete()
    if current_user.profile_pic != "images/profile/def.jpg":
        response = cloudinary.uploader.destroy(current_user.username)
        if response.get("result") != "ok":
            print("*******ALERT*********:", response)
    db.commit()
    return {"message": "User deleted successfully"}


# ----------------- Routes for confessions--------------------


# --------------------- Routes for comments-------------------------

# ------------------------------------------------------------------
@app.post("set/email/")
async def set_email(password: str):
    if password != os.getenv("PASSWORD"):
        return JSONResponse(status_code=400,
            content={"message": f"Invalid Password, 2 more attempts and your ip is blocked"})
        # IP blocking mechanism yet to be implement
    import json
    emails, names=[], []
    if not os.path.exists("data.json"):
        return HTTPException(status_code=500, detail="data.json doesn't exists")
    with open(file="data.json", encoding="utf-8", mode="r") as f:
        data = json.loads(f.read())
    mp = {}
    for student in data:
        mp[student["email"]] = student["name"]
    myKeys = list(mp.keys())
    myKeys.sort()
    mp = {i: mp[i] for i in myKeys}
    for key in mp:
        emails.append(key)
        names.append(mp[key])
    with open(file="emails.json", encoding="utf-8", mode="w") as f:
        f.write(json.dumps({"emails":emails, "names": names}))
    return JSONResponse(status_code=200, content={"message":"Success"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=2)