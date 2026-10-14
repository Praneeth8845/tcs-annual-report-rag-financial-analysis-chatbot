from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

#step1: load tcs report pdf file,store into documents
project_dir = Path(__file__).resolve().parent
loader = PyPDFLoader(str(project_dir / "documents" / "tcs.pdf"))
documents = loader.load()


print('total documents loaded:', len(documents))

#step2: split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,  
    chunk_overlap=500,
    
)
#let us load chunks into my_chunks variable
my_chunks = text_splitter.split_documents(documents)   
print('total chunks created:', len(my_chunks))

#step 3: create embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#step 4:create faiss database and store the chunks into it
index = faiss.IndexFlatL2(len(embedding_model.embed_query("hello world")))
vector_store=FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)

#step 5 : store our chunks into vector db
vector_store.add_documents(my_chunks)
print("sucessfully stored the chunks into vector store")

# step6 : store vector store[db] permenantly
vector_store.save_local("faiss_index")
print("sucessfully stored the vector store into faiss_index folder") 
''
# from real time pythonfile, is you want to use fiass we need to load faiss 
# my_vector_store = FAISS.load_local(
#     "faiss_index",
#     embedding_model,
#     allow_dangerous_deserialization=True)
# doc=new_vector_store.similarity_search("What is the revenue of TCS in 2022?", k=5)
# ''