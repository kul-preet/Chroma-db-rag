import os
import hashlib  #used to create hash of the content
from src.constants import DOCUMENTS_FOLDER
from src.fileReaders import read_file


def Load_all_docs(collection):
    '''
    Dynamically loads new documents into ChromaDB without reprocessing existing ones.
    Tracks processed files by storing file metadata and only processes new/modified files.
    '''
    print("Checking for new documents to process...\n")

    if not os.path.exists(DOCUMENTS_FOLDER):
        print(f"Documents folder '{DOCUMENTS_FOLDER}' not found. Please create it and add  your documents.")
        return

    # Get current max chunk ID to avoid conflicts
    current_count = collection.count()
    if current_count > 0:
        # Query existing IDs to find the highest chunk number
        try:
            results = collection.get(include=['ids'])
            existing_ids = [int(id.split('_')[1]) for id in results['ids'] if id.startswith('chunk_')]
            global_id = max(existing_ids) + 1 if existing_ids else 0
        except:
            global_id = current_count  # Fallback
    else:
        global_id = 0

    # Get list of all files in documents folder
    all_files = [f for f in os.listdir(DOCUMENTS_FOLDER) if not f.startswith('.')]

    # Get already processed files from collection
    processed_files = {}
    try:
        # Query for file tracking metadata
        results = collection.get(
            where={"category": "file_tracking"},
            include=['metadatas']
        )
        for metadata in results['metadatas']:
            if 'filename' in metadata and 'file_hash' in metadata:
                processed_files[metadata['filename']] = metadata['file_hash']
    except:
        pass  # No tracking data yet

    new_chunks = []
    new_metadata = []
    new_ids = []
    files_processed = 0

    # Process each file
    for filename in all_files:
        filepath = os.path.join(DOCUMENTS_FOLDER, filename)

        # Calculate file hash to detect changes
        try:
            with open(filepath, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
        except:
            print(f"  ⚠️  Could not read file: {filename}")
            continue

        # Check if file is new or modified
        if filename not in processed_files or processed_files[filename] != file_hash:
            print(f"  📄 Processing: {filename}")

            # Read and chunk the file
            file_chunks = read_file(filepath)
            if file_chunks:
                for text, metadata in file_chunks:
                    new_chunks.append(text)
                    new_metadata.append(metadata)
                    new_ids.append(f"chunk_{global_id}")
                    global_id += 1

                files_processed += 1

                # Store file tracking info
                tracking_metadata = {
                    "filename": filename,
                    "file_hash": file_hash,
                    "category": "file_tracking",
                    "processed_at": str(os.path.getmtime(filepath))
                }
                new_metadata.append(tracking_metadata)
                new_ids.append(f"tracking_{filename}")
                new_chunks.append("")  # Empty content for tracking entry
            else:
                print(f"  ⚠️  No chunks generated from: {filename}")
        else:
            print(f"  ⏭️  Skipping (unchanged): {filename}")

    # Store new chunks in batches
    if new_chunks:
        print(f"\n📥 Adding {len(new_chunks)} new chunks from {files_processed} files...")

        batch_size = 100
        for i in range(0, len(new_chunks), batch_size):
            batch_chunks = new_chunks[i:i+batch_size]
            batch_metadata = new_metadata[i:i+batch_size]
            batch_ids = new_ids[i:i+batch_size]

            collection.add(
                documents=batch_chunks,
                metadatas=batch_metadata,
                ids=batch_ids
            )
            print(f"  ✅ Stored batch {i//batch_size + 1} ({len(batch_chunks)} chunks)")

        print(f"\n🎉 Done! Added {len(new_chunks)} chunks from {files_processed} files.")
    else:
        print("  ℹ️  No new files to process.")

    total_chunks = collection.count()
    print(f"📊 Total chunks in database: {total_chunks}\n")
    
    
#--------------------RETRIEVE RELAVANT CHUNKS----------------------
def retrieve(question, collection, top_k = 3):
    #retrieve relavant chunks from chroma db based on the question\
        
    '''the chromadb convert the question to the vector using the embeding function and then 
    compare the vector and return the most similar chunks'''
    results = collection.query(
        query_texts = [question],
        n_results = top_k
    )
    
    chunks = results["documents"][0]
    metadatas = results["metadatas"][0]
    return chunks,metadatas