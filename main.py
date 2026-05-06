from src.db_setup import get_collection
from src.load_fetch_knowledge import Load_all_docs
from src.llm import ask
from src.constants import CHROMA_DB_PATH

def main():
    print("=" * 55)
    print("   📚 TechLearn RAG with ChromaDB")
    print("=" * 55 + "\n")
    #get collections
    collection = get_collection()
    #load store knowledge
    Load_all_docs(collection)

    print(f"✅ Ready! {collection.count()} chunks in database.\n")
    print("Commands:")
    print("  'quit' to exit")
    print("  'stats' to see DB info")
    print("  'reload' to check for new documents\n")

    #ask question
    while True:
        question = input("You: ").strip()

        if not question:
            continue

        if question.lower() == "quit":
            print("Goodbye!")
            break

        elif question.lower() == "stats":
            print(f"\n📊 ChromaDB Stats:")
            print(f"   Total chunks: {collection.count()}")
            print(f"   DB location: {CHROMA_DB_PATH}\n")
            continue

        elif question.lower() == "reload":
            print("\n🔄 Checking for new documents...")
            Load_all_docs(collection)
            continue

        answer = ask(question, collection)
        print(f"\nAssistant: {answer}\n")
        print("-" * 50)

        
        
if __name__ == "__main__":
    main()