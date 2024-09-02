import os
from langchain import HuggingFaceHub
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.csv_loader import CSVLoader 
import pandas as pd
from transformers import pipeline

def get_vectordb(file:str):
    filename, fileextension = os.path.splitext(file)
    embeddings = HuggingFaceEmbeddings()

    if fileextension == ".csv":
        loader = CSVLoader(file_path=file)
    elif fileextension == ".txt":
        loader = TextLoader(file_path=file)
    else:
        print("File type not supported. ")
        return None

    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap = 100
            )

    docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(docs, embeddings)

    return db

def run_llm(key,db,query:str, model)-> str:
    llm = HuggingFaceHub(
        repo_id = model,
        model_kwargs = {"temperature" : 0.8, "max_length": 1024}

    )

    prompt_temp = """
    You are an AI assistant that helps the user by providing relevant information about the document given to you.
    Use the following pieces of context to answer the question at the end. Please follow the following rules:
    1.strictly answer the question based on the given document only, no external questions must be answered
    2. if an external question is asked which is not related to given document reply with "Information not in given document"
    3. if any general knowledge question is asked to you, like the name of an animal or a country reply with "Information not in given document" 
    4. Make sure that the answer you are giving is related to the document
    5. double check the information provided to you and answer accordingly 
    6. if numerical value is relevant to the question, extract it and include accurately in your answer.
    7. there are multiple properties in the file. each property is only 1 word long. if it is numerical or alphanumeric value, then extract it.

    {context}

    Question: {question}

    Helpful Answer:
    
    """

    #now we create a retrival qa to get the info and make sure it returns source document as well
    Prompt = PromptTemplate(template= prompt_temp , input_variables=["context","question"])
    retrival = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = db.as_retriever(),
        return_source_documents = True,
        chain_type_kwargs={"prompt": Prompt}

    )

    answer =   retrival.invoke({"query":query})
    return answer

# def modify_global(path):
#     global global_csv
#     global_csv = path
#
# def get_path(path):
#     global global_csv
#     global_csv = path

def read_csv(path,query):
    df = pd.read_csv(path)
    table_data = df.astype(str).to_dict(orient="records")
    pipe = pipeline("table-question-answering", model="google/tapas-medium-finetuned-wtq")
    query.strip()
    ans = pipe(table=table_data, query=query)
    return ans
