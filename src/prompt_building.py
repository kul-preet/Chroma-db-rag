


#---------------------build prompt with context-----------------------
def build_prompt(question, chunks, metadatas):
    # Build prompt with context and souce information
    # Each chunk is labelled with its source file so that llm can site it.
    
    context_parts = []
    
    for chunk,metadata in zip(chunks, metadatas):  #zip, pairs the elements of both lists
        source = metadata.get("source", "unknown")
        file_type = metadata.get("file_type", "unknown")
        
        #add extra location info based on file type
        if file_type == "pdf":
            location = f"Page {metadata.get('page', '?')}"
        elif file_type in ["csv", "excel"]:
            location = f"Row {metadata.get('row', '?')}"
        elif file_type == "docx":
            location = f"Paragraph {metadata.get('paragraph', '?')}"
        else:
            location = f"Paragraph {metadata.get('paragraph', '?')}"
            
        # Format: [Source: policy.pdf, Page 2]
        source_label = f"[Source: {source}, {location}]"

        context_parts.append(f"{source_label}\n{chunk}")
 
        
    context = "\n\n".join(context_parts)
    
    prompt = f"""You are a helpful assistant for TechLearn India.
    Use the following context to answer the question.
    Always mention which source/file the information came from.
    If the answer is not in the context say "I don't have that information."
    Context: {context}

    Customer Question: {question}

    Answer: """
    return prompt