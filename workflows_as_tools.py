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

class ArxivRetrievalInput(BaseModel):
    text_name: str = Field(description="The name of the collection of articles")

class OcrEnhancingInput(BaseModel):
    main_text_filename: str = Field(description="The main text file to be enhanced")
    supporting_text_filename: str = Field(description="The supporting text file")
class ProofRemovalInput(BaseModel):
    main_text_filename: str = Field(description="The main text file to be translated")

class KeywordSummaryInput(BaseModel):
    main_text_filename: str = Field(description="The main text file to be summarized")

class CitationExtractorInput(BaseModel):
    main_text_filename: str = Field(description="The main text file to be summarized")
    extraction_type : str = Field(description="The type of extraction") 
    auxilary_text_filename: str = Field(description="The auxilary file with keywords and summary")

class ArxivRetrievalToolClass:
    def __init__(self, retriever_model=None, cleaner_model=None, receptionist_model=None):
        if retriever_model==None:
            self.retriever_model=ChatOpenAI(model="gpt-3.5-turbo")
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
        retrieve_app = ArxivRetrievalWorkflow(retriever_model=self.retriever_model, cleaner_model=self.cleaner_model, 
                                     receptionist_model=self.receptionist_model)
        retrieve_app=retrieve_app.create_workflow()
        retrieve_app = retrieve_app.compile()
        state = retrieve_app.invoke(input)    
        return state["receptionist_retriever_history"][-1].content

class OcrEnhancingToolClass():
    def __init__(self, enhancer_model=None, embeder=None):
        
        if enhancer_model==None:
            self.enhancer_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.enhancer_model = enhancer_model

        if embeder==None:
            self.embeder=OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            self.embeder = embeder

        self.description = """This tool takes a text in two text and improve the one using the second 
        as a reference."""
    def ocr_enhance(self, main_text_filename: str, supporting_text_filename: str) -> str:
        """This tool takes a text in two text and improve the one using the second 
        as a reference."""
        input = {
            "main_text_filename": HumanMessage(content=main_text_filename),
            "supporting_text_filename": HumanMessage(content=supporting_text_filename),
            "report": HumanMessage(content="")}    
        ocr_enchancer_app = OcrEnchancingWorkflow(enhancer_model=self.enhancer_model, embeder=self.embeder)
        ocr_enchancer_app = ocr_enchancer_app.create_workflow()
        ocr_enchancer_app = ocr_enchancer_app.compile()
        state = ocr_enchancer_app.invoke(input)
        return state["report"].content


class ProofRemovalToolClass():
    def __init__(self, stamper_model=None,remover_model=None):
        if stamper_model==None:
            self.stamper_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.stamper_model = stamper_model
        if remover_model==None:
            self.remover_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.remover_model = remover_model
        self.description = "This tool takes a text in a form of a string and removes the proof section from the text."
    
    def remove_proof(self, main_text_filename: str) -> str:
        """This tool takes a text in a form of a string and removes the proof section from the text."""
        input = {
            "main_text_filename": HumanMessage(content=main_text_filename),
            "report": HumanMessage(content=""),
            "file": [""]
        }    
        proof_remover_app =ProofRemovingWorkflow(stamper_model=self.stamper_model, remover_model=self.remover_model)
        proof_remover_app = proof_remover_app.create_workflow()
        proof_remover_app = proof_remover_app.compile()
        state = proof_remover_app.invoke(input)
        return state["report"].content

class KeywordAndSummaryToolClass:
    def __init__(self, keyword_and_summary_model=None):
        if keyword_and_summary_model==None:
            self.keyword_and_summary_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.keyword_and_summary_model = keyword_and_summary_model
        self.description="""
        This tool takes a string that corresponds to the filename of a text.
        It processes the text in order to extract keywords and summary which it puts in a file.
        It returns the report of the process."""
        
    def get_keyword_and_summary(self, main_text_filename: str) -> str:
        """This tool takes a string that corresponds to the filename of a text. 
        It processes the text in order to extract keywords and summary which it puts in a file.
        It returns the report of the process."""
        input = {
            "main_text_filename": HumanMessage(content=main_text_filename),
            "report": HumanMessage(content="")
        }
        keyword_and_summary_app = KeywordAndSummaryWorkflow()
        keyword_and_summary_app = keyword_and_summary_app.create_workflow()
        keyword_and_summary_app = keyword_and_summary_app.compile()
        state = keyword_and_summary_app.invoke(input)
        return state["report"].content


class TranslationToolClass:
    def __init__(self, translator_model=None):
        if translator_model==None:
            self.translator_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
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
        input = {"keywords_and_summary_filename": HumanMessage(content=keywords_and_summary_filename),
            "target_language": HumanMessage(content=target_language),
            "main_text_filename": HumanMessage(content=main_text_filename),
            "report": HumanMessage(content="")}
        translation_app=TranslationWorkflow(translator_model=self.translator_model)
        translation_app=translation_app.create_workflow()
        translation_app=translation_app.compile()
        state = translation_app.invoke(input)
        return state["report"].content

class CitationExtractionToolClass:
    def __init__(self, citation_extractor_model=None):
        if citation_extractor_model==None:
            self.citation_extractor_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.citation_extractor_model = citation_extractor_model
        self.description="""This tool takes three strings that correspond to the filename of a text_file from which we want to extract the citations,
            a type of extraction (all of them, the most important etc etc), and the filename of a text that can be used as a context for better extraction. 
            it extracts the citations and  saves the result as a file on the disk. It returns a report of the process."""
    def extract_citations(self, main_text_filename: str, extraction_type: str, auxilary_text_filename: str) -> str:
        input = {"main_text_filename": HumanMessage(content=main_text_filename),"extraction_type": HumanMessage(content=extraction_type),
        "auxilary_text_filename": HumanMessage(content=auxilary_text_filename),"report": HumanMessage(content="")}
        citation_extraction_app=CitationExtractionWorkflow(citation_extractor_model=self.citation_extractor_model)
        citation_extraction_app=citation_extraction_app.create_workflow()
        citation_extraction_app=citation_extraction_app.compile()
        state = citation_extraction_app.invoke(input)
        return state["report"].content
    
def create_tools():    
    
    TranslationTool=TranslationToolClass()
    TranslationTool=StructuredTool(name="TranslationTool",func=TranslationTool.translate_file,args_schema=TranslatorInput,
                           description=TranslationTool.description)
    ArxivRetrievalTool=ArxivRetrievalToolClass()
    ArxivRetrievalTool=StructuredTool(name="ArxivRetrievalTool",func=ArxivRetrievalTool.retrieve_bib,args_schema=ArxivRetrievalInput,
                           description=ArxivRetrievalTool.description)
    OcrEnhancingTool=OcrEnhancingToolClass()
    OcrEnhancingTool=StructuredTool(name="OcrEnhancingTool",func=OcrEnhancingTool.ocr_enhance,args_schema=OcrEnhancingInput)

    ProofRemoverTool=ProofRemovalToolClass()
    ProofRemoverTool=StructuredTool(name="ProofRemovalTool",func=ProofRemoverTool.remove_proof,args_schema=ProofRemovalInput)

    KeywordAndSummaryTool=KeywordAndSummaryToolClass()
    KeywordAndSummaryTool=StructuredTool(name="KeywordAndSummaryTool",func=KeywordAndSummaryTool.get_keyword_and_summary,
                                         args_schema=KeywordSummaryInput)
    tools=[TranslationTool,ArxivRetrievalTool,OcrEnhancingTool,ProofRemoverTool,KeywordAndSummaryTool, pdf_to_markdown]
    return tools


