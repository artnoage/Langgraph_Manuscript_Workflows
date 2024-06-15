import urllib, os, subprocess, pathlib, pymupdf4llm
from prompts import *
from langchain_core.tools import tool, StructuredTool
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_text_splitters import CharacterTextSplitter
from tqdm import tqdm
from langchain_core.messages import HumanMessage 
from simple_workflows import *
from langchain.pydantic_v1 import BaseModel, Field

### This file contains complex tools, which means that each tool is a workflow
### or a chain, the take two variables. One is the models that are used and the 
### other is to choose between streaming and printing.
class TranslatorInput(BaseModel):
    keywords_and_summary_filename: str = Field(description="The auxilary file with keywords and summary"                                               )  
    target_language: str = Field(description="The target language")
    main_text_filename: str = Field(description="The main text file to be translated")

class ArxivRetrieverInput(BaseModel):
    text_name: str = Field(description="The name of the collection of articles")

class OcrEnchancerInput(BaseModel):
    main_text_filename: str = Field(description="The main text file to be translated")

class ProofRemoverInput(BaseModel):
    main_text_filename: str = Field(description="The main text file to be translated")

class ArxivRetrievalToolClass:
    def __init__(self, streaming: bool = True, streamcon=None, retriever_model=None, cleaner_model=None, receptionist_model=None):
        if retriever_model==None:
            self.retriever_model==ChatOpenAI(model="gpt-3.5-turbo")
        else:
            self.retriever_model = retriever_model
        if  cleaner_model==None:
            self.cleaner_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.cleaner_model = cleaner_model
        if receptionist_model==None:
            self.receptionist_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.receptionist_model = receptionist_model
        self.streaming = streaming
        self.streamcon =None
        self.description = "This tool takes a string that contains a collection of articles and retrieves them from arXiv."    
    def retrieve_bib(self, text_name: str) -> str:
        """This tool takes a string that contains a collection of articles and retrieves them from arXiv."""
        input = {
            "receptionist_retriever_history": [HumanMessage(content="")],
            "last_action_outcome": [HumanMessage(content="")],
            "metadata": HumanMessage(content=" "),
            "article_keywords": HumanMessage(content=" "),
            "title_of_retrieved_paper": HumanMessage(content=" "),
            "should_I_clean": False
        }
        input["receptionist_retriever_history"][0] = HumanMessage(content="Please fetch me the following papers" + text_name)
        retrieve_app = ArxivRetrievalWorkflow(streaming=self.streaming, streamcon=self.streamcon, 
                                     retriever_model=self.retriever_model, cleaner_model=self.cleaner_model, 
                                     receptionist_model=self.receptionist_model)
        retrieve_app=retrieve_app.create_workflow()
        retrieve_app = retrieve_app.compile()
        state = retrieve_app.invoke(input)    
        return state["receptionist_retriever_history"][-1].content

class OcrEnhancerToolClass:
    def __init__(self, streaming: bool = True, streamcon=None, enhancer_model=None,embeder=None):
        
        if enhancer_model==None:
            self.enhancer_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.enhancer_model = enhancer_model
        self.streaming = streaming
        self.streamcon = streamcon

        self.description = "This tool takes a text in a form of a string and performs OCR on it."
    def ocr_enchancer(self, text_name: str) -> str:
        """This tool takes a text in a form of a string and performs OCR on it."""
        input = {
            "main_text_filename": HumanMessage(content=text_name),
            "report": HumanMessage(content=""),
            "file": []
        }    
        ocr_enchancer_app = OcrEnchancerWorkflow(streaming=self.streaming, streamcon=self.streamcon, 
                                                 enhancer_model=self.enhancer_model, embeder=self.embeder)
        ocr_enchancer_app = ocr_enchancer_app.create_workflow()
        ocr_enchancer_app = ocr_enchancer_app.compile()
        state = ocr_enchancer_app.invoke(input)
        return state["report"].content


class ProofRemoverTool:
    def __init__(self, streaming: bool = True, streamcon=None, stamper_model=None,remover_model=None):
        if stamper_model==None:
            self.stamper_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.stamper_model = stamper_model
        if remover_model==None:
            self.remover_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.remover_model = remover_model
        self.streaming = streaming
        self.streamcon = streamcon
        self.description = "This tool takes a text in a form of a string and removes the proof section from the text."
    def proof_remover(self, text_name: str) -> str:
        """This tool takes a text in a form of a string and removes the proof section from the text."""
        input = {
            "main_text_filename": HumanMessage(content=text_name),
            "report": HumanMessage(content=""),
            "file": []
        }    
        proof_remover_app = ProofReamovingWorkflow()
        proof_remover_app = proof_remover_app.create_workflow(streaming=self.streaming, streamcon=self.streamcon
                                                             ,stamper_model=self.stamper_model, remover_model=self.remover_model)
        proof_remover_app = proof_remover_app.compile()
        state = proof_remover_app.invoke(input)
        return state["report"].content

class KeywordAndSummaryMakerTool:
    def __init__(self, streaming: bool = True, streamcon=None, keyword_and_summary_model=None):
        if keyword_and_summary_model==None:
            self.keyword_and_summary_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.keyword_and_summary_model = keyword_and_summary_model
        self.streaming = streaming
        self.streamcon = streamcon
        self.description="""
        This tool takes a string that corresponds to the filename of a text.
        It processes the text in order to extract keywords and summary which it puts in a file.
        It returns the report of the process."""
        
    def keyword_and_summary(self, text_name: str) -> str:
        """This tool takes a string that corresponds to the filename of a text. 
        It processes the text in order to extract keywords and summary which it puts in a file.
        It returns the report of the process."""
        input = {
            "main_text_filename": HumanMessage(content=text_name),
            "report": HumanMessage(content="")
        }
        keyword_and_summary_app = KeywordAndSummaryWorkflow()
        keyword_and_summary_app = keyword_and_summary_app.create_workflow()
        keyword_and_summary_app = keyword_and_summary_app.compile()
        state = keyword_and_summary_app.invoke(input)
        return state["report"].content


class TranslationToolClass:
    def __init__(self, streaming=False, streamcon=None, translator_model=None):
        if translator_model==None:
            self.translator_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.translator_model = translator_model
        self.streaming = streaming
        self.streamcon = streamcon
        self.translator_model = translator_model 
        self.description="""
        This tool takes three strings that correspond to the filename of a text containing keywords,
        a choice of language, and the filename of a text to be translated. Then it translates it to the 
        target language and saves the result as a file on the disk. It returns a report of the process.
        """
    def translate_file(self, keywords_and_summary_filename: str, target_language: str, main_text_filename: str) -> str:
        """
        This tool takes three strings that correspond to the filename of a text containing keywords,
        a choice of language, and the filename of a text to be translated. Then it translates it to the 
        target language and saves the result as a file on the disk. It returns a report of the process.
        """
        input = {
            "keywords_and_summary_filename": HumanMessage(content=keywords_and_summary_filename),
            "target_language": HumanMessage(content=target_language),
            "main_text_filename": HumanMessage(content=main_text_filename),
            "report": HumanMessage(content=""),
        }
        translation_app=TranslationWorkflow(streaming=self.streaming, streamcon=self.streamcon,
                                            translator_model=self.translator_model)
        translation_app=translation_app.create_workflow()
        translation_app=translation_app.compile()
        state = translation_app.invoke(input)
        return state["report"].content


