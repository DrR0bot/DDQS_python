# Document Database and Query System

This repository contains a system for building and querying a database of document embeddings, designed to handle document storage, retrieval, and question-answering tasks using the `Chroma` database. The system supports dynamic updates by calculating hashes for document content and provides efficient retrieval through similarity searches using embeddings generated by various models (currently set to use **BAAI/bge-base-en-v1.5** from HuggingFace).

## Features

- **Document Ingestion**: Load PDF documents, split them into manageable chunks, and store them in a Chroma database with corresponding embeddings and metadata.
- **Hash-based Updates**: Efficiently track changes in document content via content hashing and only update changed chunks.
- **Embeddings with HuggingFace**: Leverages the `HuggingFaceBgeEmbeddings` for generating document and query embeddings.
- **Query Interface**: A command-line tool to query the database and retrieve relevant document chunks using a similarity search and an LLM-powered response system.

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/DrR0bot/DDQS_python.git
   cd DDQS_python
   ````

2. **Clone the repository**:
    Install required dependencies: This project uses Python, so ensure you have it installed. Then, install the dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ````

3. **Install PyTorch with CUDA (Optional)**:
    If you want to leverage GPU (CUDA) for faster embedding generation, install PyTorch with CUDA support:
    
    Using `pip`:
    ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    ````
    
    Using `conda`:
    If you are using Conda, install PyTorch and related libraries with CUDA support:
    ```bash
    conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
    ````

## Usage

1. **Populate the Database**
The populate_database.py script loads documents from a directory, splits them into chunks, generates embeddings, and stores them in the Chroma database. If a document is modified, only the updated chunks will be added or replaced.
    ```bash
    python populate_database.py [--reset]

`--reset`: Optional flag to clear the existing database before populating it with new data.

2. **Query the Database**
Once the documents are ingested into the database, you can run queries to retrieve relevant chunks based on their embeddings and generate a response using an LLM model.
    ```bash
    python query_data.py "Your query text here"
    ````

The system will retrieve the most similar document chunks based on the query text and generate a response using the context.

## File Structure
- populate_database.py: Handles loading PDF documents, splitting them into chunks, generating embeddings, and storing/updating them in the Chroma database.
- `get_embedding_function.py`: Contains the logic to select and configure the embedding model (currently using `HuggingFaceBgeEmbeddings`).
- `query_data.py`: Handles querying the database using a similarity search and generating responses with an LLM model.

## Configuration
**Changing the Embedding Model**
By default, the system uses the `HuggingFaceBgeEmbeddings` model (BGE Base) from HuggingFace. You can change the model or switch between using CPU and GPU in `get_embedding_function.py`.

To change the device to GPU, modify this line in get_embedding_function.py:
    ```bash
    model_kwargs = {"device": "cuda"}  # Use CUDA for GPU
    ````

To use another embedding model, change the model_name parameter:
    ```bash
    model_name = "BAAI/bge-base-en-v1.5"  # Change to your preferred model
    ````

## Modifying Document Paths
In `populate_database.py`, update the `DATA_PATH` variable to point to the directory where your PDF files are stored:
```bash
    DATA_PATH = "data"  # Change this to the path of your documents
````

## Requirements
 - Python 3.8+
 - HuggingFace Transformers
 - PyTorch (with or without CUDA)
 - LangChain Community Packages


## License
This project is licensed under the MIT License.