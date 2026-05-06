
import chromadb
from src.constants  import CHROMA_DB_PATH
from src.constants  import COLLECTION_NAME
from chromadb.utils import embedding_functions


#---------------------EMBEDDING FUNCION----------------------
# How:   We use sentence-transformers — a free local model
#        "all-MiniLM-L6-v2" is small, fast, and works well for this use case
#        It converts any text into a vector of 384 numbers
'''embedding function will convert the text into vector'''
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name = "all-MiniLM-L6-v2")


#---------------------CHROMA DB SETUP---------------------
def get_collection():
    #create a chroma db client
    chroma_client = chromadb.PersistentClient(path = CHROMA_DB_PATH)
    
    #check if collection exists, if not create it
    collection = chroma_client.get_or_create_collection(
        name = COLLECTION_NAME,
        embedding_function = embedding_fn
    )
    
    return collection