import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("omnisentinel.database")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "omnisentinel")

class Database:
    client: AsyncIOMotorClient = None
    db = None

db_instance = Database()

async def connect_to_mongo():
    try:
        db_instance.client = AsyncIOMotorClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
        # Attempt to connect to check if the server is available
        await db_instance.client.server_info()
        db_instance.db = db_instance.client[MONGODB_DATABASE]
        logger.info(f"Connected to MongoDB at {MONGODB_URI}")
    except Exception as e:
        logger.warning(f"Could not connect to MongoDB: {e}")
        db_instance.client = None
        db_instance.db = None

async def close_mongo_connection():
    if db_instance.client:
        db_instance.client.close()
        logger.info("Closed MongoDB connection")

def get_db():
    return db_instance.db
