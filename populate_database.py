import argparse
import os
import shutil
import hashlib
from datetime import datetime
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from keywords_extraction import extract_and_refine_keywords
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --reset flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()


def calculate_page_hash(page_content: str):
    # Generate a hash for the page content
    return hashlib.md5(page_content.encode('utf-8')).hexdigest()


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,  # Adjust this to target sentence/paragraph granularity
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=True,  # Allow regex-based splitting (for more control)
        separators=["\n\n", ".", "?", "!"]  # Paragraph/sentence split markers
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs and hashes.
    chunks_with_ids_and_hashes = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=["metadatas"])  # Retrieve metadata including hashes
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Create a mapping of existing document IDs to their hashes
    existing_id_to_hash = {
        existing_items["ids"][i]: existing_items["metadatas"][i]["hash"]
        for i in range(len(existing_items["ids"]))
    }

    new_chunks = []
    updated_chunks = []
    
    for chunk in chunks_with_ids_and_hashes:
        chunk_id = chunk.metadata["id"]
        chunk_hash = chunk.metadata["hash"]
        
        if chunk_id not in existing_ids:
            new_chunks.append(chunk)
        else:
            # Check if the hash has changed
            stored_chunk_hash = existing_id_to_hash.get(chunk_id)
            if stored_chunk_hash != chunk_hash:
                updated_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
    else:
        print("✅ No new documents to add")

    if len(updated_chunks):
        print(f"🔄 Updating modified documents: {len(updated_chunks)}")
        updated_chunk_ids = [chunk.metadata["id"] for chunk in updated_chunks]
        db.update_documents(updated_chunks, ids=updated_chunk_ids)
    else:
        print("✅ No documents to update")


def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id
        page_content = chunk.page_content
        chunk_hash = calculate_page_hash(page_content)
        chunk.metadata["hash"] = chunk_hash

        file_name_with_extension = os.path.basename(source)
        chunk.metadata["file_name"] = file_name_with_extension

        file_stats = os.stat(source)
        creation_date = datetime.fromtimestamp(file_stats.st_ctime).strftime('%Y-%m-%d %H:%M:%S')
        modification_date = datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')

        chunk.metadata["creation_date"] = creation_date
        chunk.metadata["modification_date"] = modification_date

        # Convert keywords list to comma-separated string
        keywords = extract_and_refine_keywords(page_content)
        chunk.metadata["keywords"] = ", ".join(keywords)  # Convert list to string

    return chunks



def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
