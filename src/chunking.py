
import re
from src.constants import CHUNK_SIZE, CHUNK_OVERLAP


#--------------------SMART TEXT CHUNKING ----------------------------
def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    '''
    splits the contionious text into overlapping chunks
    tries to split the sentance from (.) then words
    '''
    #step 1 :- clean the text
    #replace the multiple spaces/newlines with single space
    text = re.sub(r'\s+',' ',text).strip()
    
    #if text is smaller than chunk_size - return as single chunk
    if len(text) <= CHUNK_SIZE:
        return[text]
    
    chunks = []
    start = 0
    #here first find the starting and ending index in the chunk and then 
    #extract the chunk on the basis of starting and ending index
    while start < len(text):
        end = start+chunk_size
        #if we haven't reached the end yet
        if end < len(text):
            #trying to find the last sentance ending (.)
            #to prevent the text from cutting from middle
            last_period = text.rfind('.',start,end) #finding last occurence of (.) from start to end
            
            if last_period != -1  and last_period > start + (chunk_size // 2):
                #we got the text in second half
                # cut from there instead of exact chunk size
                end = last_period + 1
            else:
                last_space = text.rfind(' ',start,end)
                if last_space != -1:
                    end = last_space
        #extracting chunk
        chunk = text[start:end].strip()
        
        #only add non empty chunks
        if chunk :
            chunks.append(chunk)            
        
        #moving start bid backward for overlapping
        start = end - overlap
        
        if start >= end:
            start = end
            
            
    return chunks