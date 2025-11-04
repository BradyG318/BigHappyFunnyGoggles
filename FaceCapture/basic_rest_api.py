from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import asyncpg
import os
import uvicorn
from dotenv import load_dotenv

app = FastAPI(title="Items API", version="1.0.0")

load_dotenv()

# CORS for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Database connection
async def get_db_connection():
    conn = await asyncpg.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '5432')),
        user=os.getenv('DB_USER', 'postgres'),
        password=os.getenv('DB_PASSWORD', ''),
        database=os.getenv('DB_NAME', 'test_db')
    )
    try:
        yield conn
    finally:
        await conn.close()

# Response models
class ItemResponse(BaseModel):
    id: int
    face: List[float]

class ItemCreate(BaseModel):
    face: List[float]

class HealthResponse(BaseModel):
    status: str
    database: bool
    message: str

# Basic health check
@app.get("/health", response_model=HealthResponse)
async def health_check(conn=Depends(get_db_connection)):
    try:
        # Test database connection
        result = await conn.fetchval("SELECT 1")
        return {
            "status": "healthy",
            "database": True,
            "message": "API and database are working"
        }
    except Exception as e:
        return {
            "status": "unhealthy", 
            "database": False,
            "message": f"Database error: {str(e)}"
        }

# Get all items
@app.get("/items", response_model=List[ItemResponse])
async def get_all_items(conn=Depends(get_db_connection)):
    try:
        items = await conn.fetch("SELECT * FROM items ORDER BY id")
        # Convert vector to list
        formatted_items = []
        for item in items:
            # Parse the vector string to Python list
            vector_str = item['face'].strip('[]')
            vector_list = [float(x) for x in vector_str.split(',')]
            formatted_items.append({
                'id': item['id'],
                'face': vector_list
            })
        return formatted_items
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch items: {str(e)}"
        )

# Get item by ID
@app.get("/items/{item_id}", response_model=ItemResponse)
async def get_item(item_id: int, conn=Depends(get_db_connection)):
    try:
        item = await conn.fetchrow(
            "SELECT * FROM items WHERE id = $1", 
            item_id
        )
        if not item:
            raise HTTPException(
                status_code=404,
                detail=f"Item {item_id} not found"
            )
        
        # Parse vector to list
        vector_str = item['face'].strip('[]')
        vector_list = [float(x) for x in vector_str.split(',')]
        
        return {
            'id': item['id'],
            'face': vector_list
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch item: {str(e)}"
        )

# Create new item
@app.post("/items", response_model=ItemResponse)
async def create_item(item: ItemCreate, conn=Depends(get_db_connection)):
    try:
        # Convert list to PostgreSQL vector format
        vector_str = f"[{','.join(map(str, item.face))}]"
        
        result = await conn.fetchrow(
            """INSERT INTO items (face) 
               VALUES ($1) 
               RETURNING *""",
            vector_str
        )
        
        # Parse the returned vector
        returned_vector_str = result['face'].strip('[]')
        returned_vector_list = [float(x) for x in returned_vector_str.split(',')]
        
        return {
            'id': result['id'],
            'face': returned_vector_list
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create item: {str(e)}"
        )

# Get total item count
@app.get("/items/count")
async def get_item_count(conn=Depends(get_db_connection)):
    try:
        count = await conn.fetchval("SELECT COUNT(*) FROM items")
        return {"count": count}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get count: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("basic_rest_api:app", host="0.0.0.0", port=8000, reload=True)