
from groq import Groq
from dotenv import load_dotenv
import os
import chromadb

#---------LOAD ENV-------------
load_dotenv()

#-----------CONSTANTS----------------
KNOWLEDGE_FILE = "knowledge.txt"
CHROMA_DB_PATH =  "./chroma_db"
#-------------Chunking settings---------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

#------------COLLECTION-----------------
#This is the name of collection, same as the name of Table in SQL
DOCUMENTS_FOLDER = "./documents"
COLLECTION_NAME = "multi_file_rag"
MODEL = "llama-3.3-70b-versatile"

#---------------------GROQ CLIENT----------------------
client = Groq(api_key = os.environ.get("GROQ_API_KEY"))