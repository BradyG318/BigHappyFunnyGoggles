import asyncpg
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import asyncio

load_dotenv()

class db_link:
    def __init__(self):
        self.connection_pool = None
        self.event_loop = None
    
    def get_event_loop(self):
        """Get or create event loop for synchronous operations"""
        try:
            return asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop, create new one
            if self.event_loop is None:
                self.event_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(self.event_loop)
            return self.event_loop
    
    def close(self):
        """Close database connection"""
        if self.conn:
            loop = self.get_event_loop()
            loop.run_until_complete(self.conn.close())

    # Asynchronous methods

    async def init_connection(self):
        """Initialize database connection"""
        self.conn = await asyncpg.connect(
            host=os.getenv('DB_HOST', 'localhost'),
            port=int(os.getenv('DB_PORT', '5432')),
            user=os.getenv('DB_USER', 'postgres'),
            password=os.getenv('DB_PASSWORD', ''),
            database=os.getenv('DB_NAME', 'faceDB')
        )

        print("Database connection initialized.")
    
    async def get_all_vectors_async(self) -> Dict[int, List[float]]:
        """Get all face vectors from database"""
        rows = await self.conn.fetch('SELECT id, encoding FROM faces')
        vectors_dict = {}
        for row in rows:
            # pgvector returns the vector as a string that needs parsing
            vector_str = row['encoding']
            if vector_str:
                # Remove brackets and split by commas
                vector_list = [float(x) for x in vector_str.strip('[]').split(',')]
                vectors_dict[row['id']] = vector_list
        return vectors_dict
    
    async def save_face_vector_async(self, id: int, encoding: List[float]) -> bool:
        """Save or update face vector in database"""
        try:
            # Convert list to pgvector format: [1.0, 2.0, 3.0]
            vector_str = '[' + ','.join(map(str, encoding)) + ']'

            await self.conn.execute('''
                INSERT INTO faces (id, encoding) 
                VALUES ($1, $2)
            ''', id, vector_str)
            return True
        except Exception as e:
            print(f"Error saving vector to database: {e}")
            return False
    
    async def delete_entry_async(self, id: int) -> bool:
        """Delete a face entry by ID"""
        try:
            await self.conn.execute('DELETE FROM faces WHERE id = $1', id)
            return True
        except Exception as e:
            print(f"Error deleting entry from database: {e}")
            return False
    
    async def clear_db_async(self) -> bool:
        """Clear all entries in the faces table"""
        try:
            await self.conn.execute('DELETE FROM faces')
            print("Database cleared.")
            return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False

    async def get_face_image_async(self, id: int) -> Any:
        """Get face image path by ID and return image data"""
        try:
            row = await self.conn.fetchrow('SELECT path FROM faces WHERE id = $1', id)
            
            if row:
                image_path = row['path']
                if os.path.exists(image_path):
                    with open(image_path, 'rb') as img_file:
                        image_data = img_file.read()
                    return image_data
                else:
                    print(f"Image file does not exist: {image_path}")
                    return None
            else:
                print(f"No image path found for ID: {id}")
                return None
                
        except Exception as e:
            print(f"Error retrieving image from database: {e}")
            return None

    # Synchronous wrappers for async methods

    def initialize(self):
        """Synchronous wrapper to initialize"""
        loop = self.get_event_loop()
        loop.run_until_complete(self.init_connection())

    def get_all_vectors(self) -> Dict[int, List[float]]:
        """Synchronous wrapper to get all vectors"""
        loop = self.get_event_loop()
        return loop.run_until_complete(self.get_all_vectors_async())
    
    def save_face_vector(self, face_id: int, vector: List[float]) -> bool:
        """Synchronous wrapper to save face vector"""
        loop = self.get_event_loop()
        return loop.run_until_complete(self.save_face_vector_async(face_id, vector))

    def clear_db(self) -> bool:
        """Synchronous wrapper to clear database"""
        loop = self.get_event_loop()
        return loop.run_until_complete(self.clear_db_async())

    def delete_entry(self, id: int) -> bool:
        """Synchronous wrapper to delete entry by id"""
        loop = self.get_event_loop()
        return loop.run_until_complete(self.delete_entry_async(id))

    def get_face_image(self, id: int):
        """Synchronous wrapper to get image data by id"""
        loop = self.get_event_loop()
        return loop.run_until_complete(self.get_face_image_async())

# Global database handler instance
db_link = db_link()