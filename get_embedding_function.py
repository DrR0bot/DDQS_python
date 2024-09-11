from langchain_community.embeddings import HuggingFaceBgeEmbeddings

def get_embedding_function():
    
    model_name = "BAAI/bge-base-en-v1.5"
    model_kwargs = {"device": "cpu"}  # Adjust device settings (use "cuda" if GPU available)
    encode_kwargs = {"normalize_embeddings": True}  # Normalization for embeddings

    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    return hf