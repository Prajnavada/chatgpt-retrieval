from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter

documents = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0).split_documents(DirectoryLoader("data/docs").load())
