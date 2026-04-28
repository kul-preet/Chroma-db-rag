# RAG with Chroma DB
from groq import Groq
from dotenv import load_dotenv
import os
import chromadb
from chromadb.utils import embedding_functions



#---------------------Full project pipeline------------------------
''' 
FULL PROJECT PIPELINE:
1. get collection: create or get chroma db collection
2. load and store knowledge:  load knowledge from knowledge.txt file and separate into chunks and sotre in db
3. user ask a question
4. retrieve relavant chunks: based on the question, retrieve relavant chunks from db
5. build prompt: combime the chunks and the question to build a prompt for llm
6. generate answer: send the prompt to llm and get answer
'''


#---------LOAD ENV-------------
load_dotenv()

#-----------CONSTANTS----------------
KNOWLEDGE_FILE = "knowledge.txt"
CHROMA_DB_PATH =  "./chroma_db"

#------------COLLECTION-----------------
#This is the name of collection, same as the name of Table in SQL
COLLECTION_NAME = "techlearn_knowledge"

MODEL = "llama-3.3-70b-versatile"

#---------------------GROQ CLIENT----------------------
client = Groq(api_key = os.environ.get("GROQ_API_KEY"))


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


#--------------------LOAD AND STORE KNOWLEDGE---------------------
def load_and_store_knowledge(collection):
    #load the knowledge.txt file, separate into chunks and store in chroma db
    if collection.count() > 0:
        print(f"collection already has {collection.count()} chunks, skipping loading")
        return
    
    #loading knowledge file
    with open(KNOWLEDGE_FILE, "r") as f:
        content = f.read()
        chunks = []
        for paragraph in content.split("\n\n"):
            if paragraph.strip():
                chunks.append(paragraph.strip())
    
    print(f"loaded {len(chunks)} chunks from knowledge file")
    collection.add(
        documents = chunks,
        ids = [f"chunk_{i}" for i in range(len(chunks))]
    )
    
    print(f"Successfully stored {len(chunks)} chunks in ChromaDB!\n")
    
    
    
#--------------------RETRIEVE RELAVANT CHUNKS----------------------
def retrieve(question, collection, top_k = 2):
    #retrieve relavant chunks from chroma db based on the question\
        
    '''the chromadb convert the question to the vector using the embeding function and then 
    compare the vector and return the most similar chunks'''
    results = collection.query(
        query_texts = [question],
        n_results = top_k
    )
    
    chunks = results["documents"][0]
    return chunks


#---------------------build prompt with context-----------------------
def build_prompt(question, chunks):
    #combines the question and chunks and build a prompt for the language model
    
    context = "\n\n".join(chunks)
    
    prompt = f"""You are a helpful assistant. Use the following knowledge to answer the question. If you don't know the answer, say you don't know.

Context: {context}

Customer Question: {question}

Answer: """
    return prompt
    
    
#---------------------generate answer----------------------
def ask(question,collection):
    ''' Retrieve the relavant chunks from chroma db,
    build prompt and generate answer
    give it to  llm
    return answer'''
    print(f"Getting the relavant chunks from the ChromaDB for the question")
    relavant_chunks = retrieve(question, collection)
    
    print(f"found {len(relavant_chunks)} relavant chunks, building prompt")
    for i, chunk in enumerate(relavant_chunks):
        print(f"Chunk {i+1}: {chunk}\n")
        
        
    prompt = build_prompt(question, relavant_chunks)
    
    #sending to llm
    response = client.chat.completions.create(
        model = MODEL,
        messages =[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature = 0.3,
        max_tokens = 512
        
    )
    answer = response.choices[0].message.content
    return answer


def main():
    print("=" * 55)
    print("   📚 TechLearn RAG with ChromaDB")
    print("=" * 55 + "\n")
    #get collections
    collection = get_collection()
    #load store knowledge
    load_and_store_knowledge(collection)
    
    print("ask me anything about techlearn,")
    print("type 'quit' to exit")
    
    #ask question
    while True:
        question = input("you :").strip()
        
        if not question:
            print("Please enter a question.")
            continue
        
        if question.lower() == "quit":
            print("Goodbye!")
            break
        
        answer = ask(question,collection)
        print(f"TechLearn Bot: {answer}\n")
        print("-" * 55 + "\n")
        
        
if __name__ == "__main__":
    main()