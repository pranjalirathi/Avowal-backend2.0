import logging
import traceback
from fastapi import FastAPI, Request, Depends, UploadFile, BackgroundTasks, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import select, func, update, delete
import os
from database import get_session, init_db
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from models import Confession, User, Comment
from schema import (
    ConfessionCreate,
    ConfessionResponse,
    UserCreate,
    UserResponse,
    CommentCreate,
    CommentResponse,
    ForgotPasswordRequest,
    ResetPasswordRequest,
    MarkAsReadRequest,
)
from sqlalchemy.exc import IntegrityError
from typing import Any, AsyncGenerator, Dict, List, Optional, Union
import re
from dotenv import load_dotenv
from pathlib import Path
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse
import asyncio
import json
from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadTimeSignature
from bisect import bisect_left
import cloudinary
import cloudinary.uploader
from data import emails_list, name_list
from utils import LLM_analyzer
from config import *
from helpers import (
    delete_confession_and_related,
    get_user_by_username,
    get_user_by_email,
    create_user,
    authenticate_user,
    create_access_token,
    verify_token,
    get_current_user,
    pwd_context
)
from sqlalchemy.orm import selectinload, load_only

# -----------------------------------------Do Not Change here----------------------------------------------------------------------

app = FastAPI()


@app.on_event("startup")
async def on_startup():
    await init_db()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

conf = ConnectionConfig(
    MAIL_USERNAME=MAIL,
    MAIL_PASSWORD=MAIL_PASSWORD,
    MAIL_FROM=MAIL,
    MAIL_PORT=587,
    MAIL_SERVER="smtp.gmail.com",
    MAIL_STARTTLS=True,
    MAIL_SSL_TLS=False,  # Use either MAIL_SSL_TLS or MAIL_STARTTLS, not both
    USE_CREDENTIALS=True,
    VALIDATE_CERTS=True,
)
cloudinary.config(
    cloud_name=CLOUD_NAME, api_key=API_KEY_CLOUD, api_secret=API_SECRET, secure=True
)

# Serializer for token generation
serializer = URLSafeTimedSerializer(SECRET_KEY)

images_path = Path("images")
app.mount("/images", StaticFiles(directory=images_path), name="images")


comment_event_queue = asyncio.Queue()

analyzer = LLM_analyzer(SYSTEM_PROMPT_FOR_APPROVAL, API_KEY_GEMINI, API_KEY_OPEN_ROUTER)

# ------------------------------------------------------------------------------------------------------------------------
@app.get("/")
async def root():
    return JSONResponse(
        status_code=200,
        content={
            "message": "Welcome to Avowal Backend - API is live, go to the documentation for more information",
            "documentation": "https://avowal-backend.vercel.app/docs",
            "Developer": "Pranjali Rathi",
        })


# ----------------------------------------------Auth Routes---------------------------------
@app.post("/signup")
async def register_user(
    user: UserCreate, session: AsyncSession = Depends(get_session)
):
    user_model = await get_user_by_username(user.username, session)
    if user_model is not None:
        return JSONResponse(
            status_code=400, content={"message": "Username already taken"}
        )
    user_model = await get_user_by_email(user.email, session)
    if user_model:
        return JSONResponse(status_code=400, content={"message": f"Email already exists"})
    if user.email not in emails_list:
        return JSONResponse(
            status_code=400,
            content={
                "message": f"This email doesn't exists in our database please enter your college mail"
            },
        )
    return await create_user(user, session)


# ------------------Login Route-----------------------
# email has to be there
@app.post("/login")
async def login_for_accesstoken(
    form_data: OAuth2PasswordRequestForm = Depends(),
    session: AsyncSession = Depends(get_session),
):
    user = await authenticate_user(
        email=form_data.username, password=form_data.password, session=session  # email is considered username here
    )
    if user is None:
        raise HTTPException(status_code=401, detail=f"Incorrect password or email")
    access_token_expire = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTE)
    access_token = create_access_token(
        data={
            "id": user.id,
            "email": user.email
        }, 
        expire_delta=access_token_expire
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/verifytoken")
async def verify_user_token(
    token: str, session: AsyncSession = Depends(get_session)
):
    await verify_token(token, session)
    return {"message": "Token is Valid"}


# ------------------------Forgot Password Routes-------------------------


@app.post("/forgot-password")
async def forgot_password(
    request: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    session: AsyncSession = Depends(get_session),
):
    user = await get_user_by_email(request.email, session)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User with this email does not exist.",
        )

    # Generate a token that expires in 10 minutes
    token = serializer.dumps(user.email, salt=SALT)
    reset_link = f"https://avowal-backend.vercel.app/reset-password?token={token}"

    # Prepare the email
    message = MessageSchema(
        subject="Password Reset Request",
        recipients=[request.email],
        body=f"Click on the link to reset your password: {reset_link}",
        subtype="html",
    )

    # Send email
    fm = FastMail(conf)
    background_tasks.add_task(fm.send_message, message)

    return {"message": "Password reset email has been sent."}


@app.post("/reset-password")
async def reset_password(
    request: ResetPasswordRequest, session: AsyncSession = Depends(get_session)
):
    try:
        # Decode the token (expires in 10 minutes)
        email = serializer.loads(request.token, salt=SALT, max_age=600)
    except SignatureExpired:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="The token has expired."
        )
    except BadTimeSignature:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid token."
        )

    # Fetch user and update the password
    user = await get_user_by_email(email, session)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="User not found."
        )

    # Update the password (hash the password as per your application logic)
    user.hashedpassword = pwd_context.hash(
        request.new_password
    )  # Make sure to hash this password before storing
    session.add(user)
    await session.commit()
    return {"message": "Password has been reset successfully."}


# -------------------------Routes for user profile--------------------------------------


# User can only update username and relationship_status
@app.put("/update")
async def update_user(
    username: Optional[str] = None,
    relationship_status: Optional[str] = None,
    session: AsyncSession = Depends(get_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    current_user = await get_user_by_email(current_user.get("email"), session)

    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")
    if relationship_status:
        if relationship_status not in ["Single", "Committed"]:
            raise HTTPException(
                status_code=400,
                detail=f"Relationship status can be either Single or Committed",
            )
        current_user.relationship_status = relationship_status
    if username:
        # Check if username is already taken (heavy db call, use redis)
        user = await get_user_by_username(username, session)
        if user and user.id != current_user.id:
            raise HTTPException(status_code=403, detail=f"Username already taken")
        current_user.username = username

    if username or relationship_status:
        session.add(current_user)
        await session.commit()
        await session.refresh(current_user)

    data = jsonable_encoder(
        current_user, include=["id", "username", "relationship_status", "name"]
    )
    return {"message": "User updated successfully", "data": data}


# User can only update profile pic
@app.post("/update_profile_pic")
async def upload_profile_pic(
    file: UploadFile,
    session: AsyncSession = Depends(get_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
):

    current_user = await get_user_by_email(current_user.get("email"), session)

    if "image" not in file.content_type:
        return JSONResponse(
            status_code=404, content={"message": "File provided is not an image"}
        )

    # delete previous file from cloud
    if current_user.profile_pic != "images/profile/def.jpg":
        cloudinary.uploader.destroy(current_user.username)

    filecontent = await file.read()

    ext = file.filename.split(".")[-1]
    if ext not in ["jpg", "jpeg", "png"]:
        return JSONResponse(
            status_code=404, content={"message": "File provided is not an image"}
        )

    upload_result = cloudinary.uploader.upload(
        filecontent,
        public_id=current_user.username,
        eager=[
            {
                "width": 500,
                "height": 500,
                "crop": "thumb",
                "gravity": "auto",
                "aspect_ratio": "1.0",
                "radius": 10,
            }
        ],
    )
    current_user.profile_pic = upload_result["eager"][0]["secure_url"]
    session.add(current_user)
    await session.commit()
    return JSONResponse(
        status_code=200,
        content={
            "message": "Profile pic updated",
            "data": {"url": upload_result["eager"][0]["secure_url"]},
        },
    )


@app.get("/profile_data")
async def get_profile(current_user: Dict[str, Any] = Depends(get_current_user), session: AsyncSession = Depends(get_session)):
    current_user = await get_user_by_email(current_user.get("email"), session)
    data = jsonable_encoder(
        current_user, exclude=["hashedpassword", "id", "unread_confessions"]
    )
    return {"message": "Profile fetched successfully", "data": data}


@app.get("/search_users")
async def search_users(
    q: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):

    stmt = select(User).where(
        (User.username.ilike(f"{q}%")) | (User.email.ilike(f"{q}%"))
    )
    results = await session.execute(stmt)
    users = results.scalars().all()

    data = [
        jsonable_encoder(user, exclude=["hashedpassword", "id", "unread_confessions"])
        for user in users
    ]
    return {"message": "Users found", "data": data}


@app.get("/user")  # viewed profile function yet to be implemented
async def get_user(
    username: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    
    user = await get_user_by_username(username, session)
    if not user:
        raise HTTPException(status_code=403, detail=f"Username not found")
    data = jsonable_encoder(user, exclude=["hashedpassword", "id", "unread_confessions"])

    return {"message": "User found", "data": data}


@app.delete("/delete_user")
async def delete_user(
    session: AsyncSession = Depends(get_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    current_user: User = await get_user_by_email(current_user.get("email"), session)
    if current_user.profile_pic != "images/profile/def.jpg":
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None, lambda: cloudinary.uploader.destroy(current_user.username)
        )
        if response.get("result") != "ok":
            logging.warning("*******ALERT*********:", response)

    await session.delete(current_user)
    await session.commit()
    return {"message": "User deleted successfully"}


# ----------------- Routes for confessions--------------------


async def extract_mentions(
    content: str, 
    session: AsyncSession
) -> Dict[str, Union[int, None]]:
    # Extract mentions using regex
    pattern = r"@(\w+)"  # Matches @username
    mentions = re.findall(pattern, content)

    res = {}
    if not mentions:
        return res

    stmt = select(User.username, User.id).where(User.username.in_(mentions))
    result = await session.execute(stmt)
    existing_users = result.all()

    existing_usernames = {user.username: user.id for user in existing_users}

    for username in mentions:
        res[username] = existing_usernames.get(username)
    return res


@app.post("/confessions", response_model=ConfessionResponse)
async def add_confession(
    confession: ConfessionCreate,
    current_user: Dict[str, Any] = Depends(get_current_user),
    session: AsyncSession = Depends(get_session),
):
    
    mentioned_usernames = await extract_mentions(confession.content, session)
    # return JSONResponse(mentioned_usernames, status_code=200)
    
    valid_user_ids = []
    for username, user_id in mentioned_usernames.items():
        if user_id is None:
            # Not to raise exception here, store the mentions as a plain text then
            # Edge case : In future with same username a user got registered in that case no linking would happen for this confession
            continue

        valid_user_ids.append(user_id)
    
    # Backward compatibility
    valid_users = await session.execute(select(User).where(User.id.in_(valid_user_ids)))
    valid_users = valid_users.scalars().all()
    
    try:
        # Check with LLM analyzer
        llm_decision = await analyzer.analyze_confession(confession.content)
    except Exception as e:
        logging.error(f"Error occurred in add_confession: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail="Error in content analysis")
    
    if llm_decision.lower().startswith("reject"):
        raise HTTPException(status_code=403, detail="Confession rejected by content analyzer")
    

    db_confession = Confession(content=confession.content, mentions=valid_users)
    session.add(db_confession)
    await session.commit()
    await session.refresh(db_confession)

    # Load mentions to avoid serialization issues and limit fields
    stmt = (
        select(Confession)
        .where(Confession.id == db_confession.id)
        .options(
            selectinload(Confession.mentions).load_only(User.id, User.username, User.profile_pic)
        )
    )
    result = await session.execute(stmt)
    db_confession: Confession = result.scalar_one()

    # This part is inefficient and should be done in a background task for production
    # For now, keeping it simple
    update_stmt = update(User).values(
        unread_confessions=func.array_prepend(
            db_confession.id, 
            User.unread_confessions
        )
    )
    await session.execute(update_stmt)
    await session.commit()

    return ConfessionResponse.from_orm(db_confession)


@app.post("/confessions/sse")  # Changed route slightly to indicate SSE
async def add_confession_sse(
    confession_data: ConfessionCreate,
    request: Request,  # Request object is needed for client disconnect check
    session: AsyncSession = Depends(get_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Receives a confession, analyzes it using an LLM via streaming,
    and saves it only if approved. Streams the process via SSE.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generates SSE messages for the analysis and saving process."""
        llm_decision = None
        analysis_stream = None
        valid_users = []

        try:
            # 1. Initial Validation (Mentions)
            yield f"event: status\ndata: {json.dumps({'message': 'Validating mentions...'})}\n\n"
            mentioned_usernames = await extract_mentions(
                confession_data.content, session
            )
            for username, user_id in mentioned_usernames.items():
                if user_id is None:
                    yield f"event: result\ndata: {json.dumps({'status': 'rejected', 'reason': f'User @{username} not found.'})}\n\n"
                    return
                result = await session.execute(select(User).where(User.id == user_id))
                valid_users.append(result.scalar_one())

            yield f"event: status\ndata: {json.dumps({'message': 'Mentions validated. Starting content analysis...'})}\n\n"

            # 2. Stream LLM Analysis
            analysis_stream = analyze_confession_with_llm(confession_data.content)
            async for chunk in analysis_stream:
                if await request.is_disconnected():
                    print("Client disconnected during analysis stream.")
                    break
                yield f"event: {chunk.get('type', 'message')}\ndata: {json.dumps(chunk)}\n\n"
                if chunk.get("type") == "decision":
                    llm_decision = chunk.get("message")

            # 3. Process Based on LLM Decision
            if llm_decision == "APPROVE":
                yield f"event: status\ndata: {json.dumps({'message': 'Content approved. Saving confession...'})}\n\n"
                try:
                    db_confession = Confession(
                        content=confession_data.content, mentions=valid_users
                    )
                    session.add(db_confession)
                    await session.commit()
                    await session.refresh(db_confession)

                    # Load mentions to avoid serialization issues and limit fields
                    stmt = (
                        select(Confession)
                        .where(Confession.id == db_confession.id)
                        .options(
                            selectinload(Confession.mentions).load_only(User.id, User.username, User.name, User.profile_pic, User.relationship_status)
                        )
                    )
                    result = await session.execute(stmt)
                    db_confession = result.scalar_one()

                    update_stmt = update(User).values(
                        unread_confessions=func.array_prepend(
                            db_confession.id, User.unread_confessions
                        )
                    )
                    await session.execute(update_stmt)
                    await session.commit()

                    yield f"event: status\ndata: {json.dumps({'message': 'Confession saved and notifications updated.'})}\n\n"
                    response_data = ConfessionResponse.from_orm(db_confession)
                    yield f"event: result\ndata: {json.dumps({'status': 'approved', 'confession': jsonable_encoder(response_data)})}\n\n"

                except Exception as e:
                    await session.rollback()
                    print(f"Unexpected error during save: {e}")
                    yield f"event: error\ndata: {json.dumps({'message': 'An unexpected error occurred during saving.'})}\n\n"
                    yield f"event: result\ndata: {json.dumps({'status': 'failed', 'reason': 'Internal server error.'})}\n\n"

            elif llm_decision == "REJECT":
                yield f"event: status\ndata: {json.dumps({'message': 'Content rejected by analysis.'})}\n\n"
                yield f"event: result\ndata: {json.dumps({'status': 'rejected', 'reason': 'Content did not meet guidelines.'})}\n\n"
            else:
                yield f"event: error\ndata: {json.dumps({'message': 'Invalid decision from analysis module.'})}\n\n"
                yield f"event: result\ndata: {json.dumps({'status': 'failed', 'reason': 'Internal analysis error.'})}\n\n"

        except Exception as e:
            print(f"Unexpected error in event generator: {e}")
            yield f"event: error\ndata: {json.dumps({'message': 'An unexpected server error occurred.'})}\n\n"
            yield f"event: result\ndata: {json.dumps({'status': 'failed', 'reason': 'Internal server error.'})}\n\n"
        finally:
            if analysis_stream:
                await analysis_stream.aclose()

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/confessions")
async def get_confessions(
    q: Optional[str] = None,
    skip: int = 0,
    limit: int = 10,
    session: AsyncSession = Depends(get_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    current_user:User = await get_user_by_email(current_user.get("email"), session)
    stmt = select(Confession)
    if q:
        stmt = stmt.where(Confession.content.ilike(f"%{q}%"))

    stmt = stmt.order_by(Confession.created_at.desc()).offset(skip).limit(limit)
    result = await session.execute(stmt)
    confessions: List[Confession] = result.scalars().all()

    unread_confessions_set = set(current_user.unread_confessions)

    data = [
        {
            "id": confession.id,
            "content": confession.content,
            "created_at": confession.created_at.isoformat(),
            "read": confession.id not in unread_confessions_set,
        }
        for confession in confessions
    ]

    return {
        "message": "Confessions fetched successfully", 
        "data": data
    }


@app.delete("/delete/confession")
async def delete_confession(
    confession_id: int,
    password: str,
    session: AsyncSession = Depends(get_session),
):
    """
    Only for admin to delete any confession
    """
    if password == os.getenv("PASSWORD"):
        result = await delete_confession_and_related(
            confession_id=confession_id,session=session) # Error handeling is not proper
        if result:
            return {"message": "Confession deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Confession not found")
    raise HTTPException(status_code=401, detail="You are not allowed here")


@app.post("/confessions/mark_as_read")
async def mark_confessions_as_read(
    request: MarkAsReadRequest,
    session: AsyncSession = Depends(get_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    user = await session.get(User, current_user.get("id"))
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    unread_confessions_set = set(user.unread_confessions)
    unread_confessions_set -= set(request.confession_ids)
    user.unread_confessions = list(unread_confessions_set)

    session.add(user)
    await session.commit()
    return {"message": "Selected confessions marked as read successfully"}


# --------------------- Routes for comments-------------------------


# Function to publish comment events
async def publish_comment_event(comment_data: dict):
    await comment_event_queue.put(comment_data)

# Changed
@app.post("/confessions/{confession_id}/comments", response_model=CommentResponse)
async def add_comment(
    confession_id: int,
    comment: CommentCreate,
    session: AsyncSession = Depends(get_session),
    current_user: Dict = Depends(get_current_user),
):
    result = await session.execute(
        select(Confession.id).where(Confession.id == confession_id)
    )
    db_confession = result.scalar_one_or_none()
    if not db_confession:
        raise HTTPException(status_code=404, detail="Confession not found.")

    db_comment = Comment(
        content=comment.content,
        user_id=current_user.get("id"),
        confession_id=confession_id
    )
        
    session.add(db_comment)
    await session.commit()

    stmt = select(Comment).where(Comment.id == db_comment.id).options(selectinload(Comment.user))
    result = await session.execute(stmt)
    db_comment = result.scalar_one()

    return CommentResponse.from_orm(db_comment)


# Can be optimized with pagination if needed
# Changed
@app.get("/comment/{confession_id}")
async def get_comments(
    confession_id: int, session: AsyncSession = Depends(get_session)
):
    stmt = select(Comment.content, Comment.user_id, Comment.id, Comment.created_at).where(Comment.confession_id == confession_id)
    result = await session.execute(stmt)
    comments = result.mappings().all()
    comments = jsonable_encoder(comments)
   
    stmt = select(User.id, User.username, User.profile_pic).where(User.id.in_([comment["user_id"] for comment in comments]))
    result = await session.execute(stmt)
    users = result.mappings().all()  # value is a dict
    user_map = {user.id: user for user in users}

    for comment in comments:
        if comment.get("user_id") in user_map:
            comment["user"] = user_map[comment.get("user_id")]
    
    return JSONResponse(status_code=200, content={"message": jsonable_encoder(comments)})


@app.delete("/comments/{comment_id}")
async def delete_comment(
    comment_id: int,
    session: AsyncSession = Depends(get_session),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    result = await session.execute(select(Comment).where(Comment.id == comment_id))
    comment = result.scalar_one_or_none()

    if not comment:
        raise HTTPException(status_code=404, detail="Comment not found")

    if current_user.get("id") != comment.user_id:
        return JSONResponse(
            status_code=400, content={"message": f"You are not allowed here"}
        )
    
    await session.delete(comment)
    await session.commit()
    
    return JSONResponse(
        status_code=200, content={"message": "Comment deleted successfully"}
    )


@app.post("set/email/")
async def set_email(password: str):
    if password != os.getenv("PASSWORD"):
        return JSONResponse(
            status_code=400,
            content={
                "message": f"Invalid Password, 2 more attempts and your ip is blocked"
            },
        )
        # IP blocking mechanism yet to be implement
    import json

    emails, names = [], []
    if not os.path.exists("data.json"):
        raise HTTPException(status_code=500, detail="data.json doesn't exists")
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
        f.write(json.dumps({"emails": emails, "names": names}))
    return JSONResponse(status_code=200, content={"message": "Success"})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, workers=2)