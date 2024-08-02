# from dotenv import load_dotenv

# load_dotenv()

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
import tempfile
import os

#Stream 출력되는 Handler
class StreamHandler(BaseCallbackHandler):
  def __init__(self, container, initial_text=""):
    self.container = container
    self.text = initial_text
  
  def on_llm_new_token(self, token: str, **kwargs) -> None:
    self.text += token
    self.container.markdown(self.text)

#제목
st.title("ChatPDF")
st.write("---")

# OpenAI API Key 입력
openai_api_key = st.text_input("OPEN_AI_API_KEY", type='password')

#파일 업로드
uploaded_file = st.file_uploader("PDF 파일을 올려주세요.", type=['pdf'])
st.write("---")

def pdf_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = PyPDFLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

#업로드 되면 동작하는 코드
if uploaded_file is not None:
    pages = pdf_to_document(uploaded_file)
   
    #Split
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 200,
        chunk_overlap  = 20,
        length_function = len,
        is_separator_regex = False,
    )
    texts = text_splitter.split_documents(pages)    

    #Embedding
    embeddings_model = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # load it into Chroma
    db = Chroma.from_documents(texts, embeddings_model)

    #Question
    st.header("PDF에게 질문해보세요!!")
    question = st.text_input('질문을 입력하세요')
    
    # if st.button('질문하기'):
    #     with st.spinner('정보를 찾는 중...'):
    #         llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key)
    #         qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
    #         result = qa_chain({"query": question})
    #         print(result)
    #         st.write(result["result"])
            
    if st.button('질문하기'):
        with st.spinner('정보를 찾는 중...'):
          chat_box = st.empty()
          stream_handler =  StreamHandler(chat_box)
          
          llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, openai_api_key=openai_api_key, streaming=True, callbacks=[stream_handler])
          qa_chain = RetrievalQA.from_chain_type(llm,retriever=db.as_retriever())
          qa_chain({"query": question})
