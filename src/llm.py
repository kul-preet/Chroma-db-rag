
from src.prompt_building import build_prompt
from src.load_fetch_knowledge import retrieve
from src.constants import MODEL,client

#---------------------generate answer----------------------
def ask(question,collection):
    ''' Retrieve the relavant chunks from chroma db,
    build prompt and generate answer
    give it to  llm
    return answer'''
    print(f"Getting the relavant chunks from the ChromaDB for the question")
    chunks,metadatas = retrieve(question, collection)
    
    for i, (chunk,meta) in enumerate(zip(chunks,metadatas)):
        print(f"  [{i+1}] {meta['source']} | {chunk[:50]}...")
        
        
    prompt = build_prompt(question, chunks, metadatas)
    
    #sending to llm
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for TechLearn India. Always cite your sources."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3,
        max_tokens=512
    )
    answer = response.choices[0].message.content
    return answer