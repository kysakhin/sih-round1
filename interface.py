import os
import streamlit as st
import pandas as pd
from backend.main import get_vectordb , read_csv, run_llm

st.title("RAG BASED AI APPLICATION FOR GROUND-WATER ASSESMENT")
st.subheader(" Works on a .txt file or a .csv file using R.A.G")
st.text("make sure that the csv file and text file are well strucutured \nand only contain well spaced characters\nyou may also get an error if your API key is invalid\nthe model may take a while to generate responses")
API_key = st.sidebar.text_input("please enter the HuggingFace api key",type= "password")

uploaded_file = st.file_uploader("UPLOAD A CSV FILE OR A TXT",type = (".txt",".csv"))
question = st.text_input(
    "Please ask your query related to your data file",
    placeholder = "example : what is the groundwater level of district BHADRADRI ?",
    disabled =not uploaded_file or not API_key
)

os.environ['HUGGINGFACEHUB_API_TOKEN'] = API_key

# models = [
#         ("google/flan-t5-large", "High performance"),
#         ("google/flan-t5-base", "Balanced"),
#         ("google/flan-t5-small", "Lightweight")
#         ]
#
# formatted_options = [f"{model} - {comment}" for model, comment in models]

option = st.selectbox("Choose what model to use? ",
                      ["google/flan-t5-large - High performance", "google/flan-t5-base - Balanced", "google/flan-t5-small - Lightweight"]
                      )

def load_data(path, nrows):
    data = pd.read_csv(path, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

if uploaded_file is not None:
    tmp = os.path.join(".", uploaded_file.name)

    with open(tmp, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write(load_data(uploaded_file.name, 10))

def filter(question):
    # to replace special characters with empty character to make splitting better.
    chars = ",./;'?[]<>\""
    for char in chars:
        question = question.replace(char, "")
    words = question.split()
    for word in words:
        if word == "mean" or word == "meaning" or word == "define" or word == "definition":
            return True
    return False

inform = "./data.txt"
if uploaded_file is not None:
    vectordb = get_vectordb(inform)
    csvpath = uploaded_file.name
    if uploaded_file is None:
        st.error("either file is not uploaded or the file type is not supported ")
    # elif vectordb is None:
    #     st.error("the file type is not supported ")
else:
     st.error("Either the file is not uploaded or the file type is not supported.")

#doing the spin thingy while generating a response
with st.spinner("Generating response..."):
    if API_key and question:
        if filter(question):
            answer = run_llm(key= API_key, db= vectordb , query= question, model=(option.split(" - ")[0]))
            st.write("### Answer")
            st.write(f"{answer['result']}")
            st.write("### Relevant source")
            rel_docs = answer['source_documents']
            for i, doc in enumerate(rel_docs):
                st.write(f"**{i+1}**: {doc.page_content}\n")
        else:
            answer = read_csv(path=tmp, query=question)
            st.write("### Answer")
            st.write(f"{answer['answer']}")

    else:
        st.info("_______________________")
