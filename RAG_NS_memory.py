from langchain_community.document_loaders import PyPDFLoader , DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load documentation

DOCUMENTS_PATH = "documents"
def load_pdf_files(documents):
    loader = DirectoryLoader(documents, glob = '*.pdf', loader_cls = PyPDFLoader)
    documents_loaded = loader.load()
    return documents_loaded

documents = load_pdf_files(documents = DOCUMENTS_PATH)
print(" Length of pdf pages ", len(documents))


# create chunks for evry documentation

def create_documents_chunks(documents_loaded):
    knowledge_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)
    knowledge_chunks = knowledge_splitter.split_documents(documents_loaded)

    return knowledge_chunks

knowledge_chunks = create_documents_chunks(documents_loaded=documents)
print(" Length of knowledge chunks ", len(knowledge_chunks))



# Create vectors embeddings documents

def get_embeddings_documents_model():
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings_model
embeddings_model = get_embeddings_documents_model()


#  store embeddings documents in FAISS 
DB_MEMORY_FAISS_PATH = "vectorStore/DB_Memory_faiss"
db = FAISS.from_documents(knowledge_chunks, embeddings_model)
db.save_local(DB_MEMORY_FAISS_PATH)
