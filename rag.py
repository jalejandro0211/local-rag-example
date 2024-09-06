from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain.vectorstores.utils import filter_complex_metadata
from utils import strip_accents

from langchain.globals import set_verbose
set_verbose(False) # 


#Para limpiar de caracteres extraños 


class ChatPDF:
    vector_store = None
    retriever = None
    chain = None

    def __init__(self):
        self.model = ChatOllama(model="qwen2:1.5b", temperature=0)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=100, )
        self.prompt = PromptTemplate.from_template(
            """
            <s> [INST] Eres un asistente para tareas de respuesta a preguntas. Usa los siguientes fragmentos de contexto 
            recuperado para responder la pregunta. Si no sabes la respuesta, simplemente di que no la sabes.
            SIEMPRE RESPONDE EN ESPAÑOL.
            Usa un máximo de tres oraciones y mantén la respuesta concisa. [/INST] </s> 
            [INST] Question: {question} 
            Context: {context} 
            Answer: [/INST]
            """
        )

    def ingest(self, pdf_file_path: str):
        docs = PyPDFLoader(file_path=pdf_file_path).load()
        chunks = self.text_splitter.split_documents(docs)
        chunks = filter_complex_metadata(chunks)
        # print(chunks)
        # chunks = strip_accents(chunks)
        # print(chunks)
        # with open("resultado.txt", "w") as archivo:
        #      for item in chunks:
        #          archivo.write(f"{item}\n")
        # #         archivo.write(chunks)
        #print(chunks)
        #vector_store = Chroma.from_documents(documents=chunks, embedding=FastEmbedEmbeddings())
        vector_store = Chroma.from_documents(
            chunks, 
            embedding=FastEmbedEmbeddings(), 
            ids=None, 
            collection_name="langchain", 
            persist_directory="./chroma_db"
        )

        self.retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 3,
                "score_threshold": 0.5,
            },
        )

        self.chain = ({"context": self.retriever, "question": RunnablePassthrough()}
                      | self.prompt
                      | self.model
                      | StrOutputParser())

    def ask(self, query: str):
        if not self.chain:
            return "Please, add a PDF document first."

        return self.chain.invoke(query)

    def clear(self):
        self.vector_store = None
        self.retriever = None
        self.chain = None

