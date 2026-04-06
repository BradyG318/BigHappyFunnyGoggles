import asyncpg
import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import asyncio
import faiss
import numpy as np

load_dotenv()

class DB_Link:
    def __init__(self):
        self.connection_pool = None
        self.conn = None
        self.event_loop = None
        self.faiss_index = None
        self.id_to_index = {} # Mapping from database ID to FAISS index
        self.index_to_id = [] # Mapping from FAISS index to database ID
    
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

    def build_faiss_index(self, vectors_dict: Dict[int, List[float]]):
        """
        Build a FAISS index from all vectors stored in the database.
        vectors_dict: {id: [float, ...]} - all normalized vectors.
        """
        if not vectors_dict:
            self.faiss_index = None
            self.id_to_index = {}
            self.index_to_id = []
            return

        dim = len(next(iter(vectors_dict.values())))   # embedding dimension
        # Use inner product index (cosine similarity on normalized vectors)
        index = faiss.IndexFlatIP(dim)

        # Prepare data
        ids = []
        vectors = []
        for db_id, vec in vectors_dict.items():
            ids.append(db_id)
            vectors.append(vec)

        vectors_np = np.array(vectors).astype('float32')

        index.add(vectors_np)   # adds vectors to index

        self.faiss_index = index
        self.index_to_id = ids
        self.id_to_index = {db_id: i for i, db_id in enumerate(ids)}
    
    def search_faiss(self, query_vector: List[float], threshold: float = 0.75, k: int = 1):
        """
        Search the FAISS index for the closest match.
        Returns (id, similarity) if similarity >= threshold, else None.
        """
        if self.faiss_index is None:
            return None

        # Prepare query
        query = np.array([query_vector]).astype('float32')

        # Search
        similarities, indices = self.faiss_index.search(query, k)
        best_sim = similarities[0][0]
        best_idx = indices[0][0]

        if best_sim >= threshold and best_idx != -1:
            return self.index_to_id[best_idx], best_sim
        return None
    
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
        rows = await self.conn.fetch('SELECT id, encoding FROM encodings') # change back to 'faces', if needed
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
        
    async def save_encoding_async(self, encoding: List[float], path: str) -> bool:
        """Save face vector and image url to encodings table"""
        try:
            # Convert list to pgvector format: [1.0, 2.0, 3.0]
            vector_str = '[' + ','.join(map(str, encoding)) + ']'

            await self.conn.execute('''
                INSERT INTO encodings (encoding, path) 
                VALUES ($1, $2)
            ''', vector_str, path)
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
    
    # Commented out because I'm paranoid
    # async def clear_db_async(self) -> bool:
    #     """Clear all entries in the faces table"""
    #     try:
    #         await self.conn.execute('DELETE FROM faces')
    #         print("Database cleared.")
    #         return True
    #     except Exception as e:
    #         print(f"Error clearing database: {e}")
    #         return False

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

    async def get_info_by_id_async(self, id: int) -> Dict[str, Any]:
        """Get all information for a face entry by ID"""
        try:
            row = await self.conn.fetchrow('SELECT * FROM info WHERE id = $1', id)
            if row:
                return dict(row)
            else:
                print(f"No entry found for ID: {id}")
                return {}
        except Exception as e:
            print(f"Error retrieving info from database: {e}")
            return {}
        
    async def get_all_paths_async(self) -> Dict[int, str]:
        """Get all image paths from the database"""
        try:
            rows = await self.conn.fetch('SELECT id, path FROM encodings')
            return {row['id']: row['path'] for row in rows if row['path']}
        except Exception as e:
            print(f"Error retrieving paths from database: {e}")
            return {}

    async def replace_encoding_async(self, id: int, new_encoding: List[float]) -> bool:
        """Replace an existing encoding by ID"""
        try:
            vector_str = '[' + ','.join(map(str, new_encoding)) + ']'
            await self.conn.execute('UPDATE encodings SET encoding = $1 WHERE id = $2', vector_str, id)
            return True
        except Exception as e:
            print(f"Error replacing encoding in database: {e}")
            return False

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

    def save_encoding(self, encoding: List[float], path: str) -> bool:
        """Synchronous wrapper to save encoding and path"""
        loop = self.get_event_loop()
        return loop.run_until_complete(self.save_encoding_async(encoding, path))

    # def clear_db(self) -> bool:
    #     """Synchronous wrapper to clear database"""
    #     loop = self.get_event_loop()
    #     return loop.run_until_complete(self.clear_db_async())

    def delete_entry(self, id: int) -> bool:
        """Synchronous wrapper to delete entry by id"""
        loop = self.get_event_loop()
        return loop.run_until_complete(self.delete_entry_async(id))

    def get_face_image(self, id: int):
        """Synchronous wrapper to get image data by id"""
        loop = self.get_event_loop()
        return loop.run_until_complete(self.get_face_image_async(id))
    
    def get_info_by_id(self, id: int) -> Dict[str, Any]:
        """Synchronous wrapper to get info by id"""
        loop = self.get_event_loop()
        return loop.run_until_complete(self.get_info_by_id_async(id))
    
    def get_all_paths(self) -> Dict[int, str]:
        """Get all image paths from the database"""
        loop = self.get_event_loop()
        return loop.run_until_complete(self.get_all_paths_async())
    
    def replace_encoding(self, id: int, new_encoding: List[float]) -> bool:
        """Synchronous wrapper to replace encoding by id"""
        loop = self.get_event_loop()
        return loop.run_until_complete(self.replace_encoding_async(id, new_encoding))

# Global database handler instance
db_link = DB_Link()