from langgraph.graph import END, StateGraph
from langchain_core.messages import  BaseMessage, ToolMessage, HumanMessage
from langgraph.prebuilt import ToolInvocation
from langgraph.prebuilt.tool_executor import ToolExecutor 
from typing import TypedDict, Annotated
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
import operator
from tqdm import tqdm
from prompts import *
from simple_tools import *
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import util
from langchain_openai import OpenAIEmbeddings
import torch
openai_api_key = os.getenv('OPENAI_API_KEY')
nvidia_api_key = os.getenv('NVIDIA_API_KEY')

class ArxivState(TypedDict):
    receptionist_retriever_history : Annotated[list[BaseMessage], operator.add]
    last_action_outcome:Annotated[list[BaseMessage], operator.add] 
    metadata: BaseMessage
    article_keywords: BaseMessage
    title_of_retrieved_paper: BaseMessage
    should_I_clean: bool
    history_reset_counter: int

class OcrEnchancerState(TypedDict):
    main_text_filename: BaseMessage
    supporting_text_filename: BaseMessage
    report:BaseMessage

class ProofRemoverState(TypedDict):
    main_text_filename: BaseMessage
    file : list[str]
    report: BaseMessage

class KeywordSummaryState(TypedDict):
    main_text_filename: BaseMessage
    report: BaseMessage

class TranslatorState(TypedDict):
    keywords_and_summary_filename: BaseMessage
    target_language: BaseMessage
    main_text_filename: BaseMessage
    report:BaseMessage
    

class ArxivRetrievalWorkflow:
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
        
        self.tools = [get_id_from_url,download_pdf]
        self.retriever=arxiv_retriever_prompt_template | self.retriever_model.bind_tools(self.tools)
        self.cleaner= arxiv_metadata_scraper_prompt_template |self.cleaner_model
        self.receptionist=arxiv_receptionist_prompt_template |self.receptionist_model
        self.tool_executor=ToolExecutor(self.tools)
    
    def run_receptionist(self,state):
        action = self.receptionist.invoke(state)
        if "We are done" in action.content:
            pr="Receptionist"+action.content
            print(pr)
        else:
            print("Receptionist: The following has been forwarded to the arxiv_retriever: ", action.content)
        return {"receptionist_retriever_history":[action],"article_keywords":action.content,"last_action_outcome":["No action was taken"], "history_reset_counter": len(state["last_action_outcome"])}
    

    def run_retriever(self,state):
        state["last_action_outcome"] = state["last_action_outcome"][state["history_reset_counter"]:]
        action = self.retriever.invoke(state) 

        if "tool_calls" in action.additional_kwargs:
            pr="Retriever: I am going to call  "+ action.tool_calls[0]["name"]
            print(pr)
            return {"last_action_outcome":[action]}
        else:
            pr="Retriever:I am reporting back to the arxiv_receptionist with"+ action.content
            print(pr)
            return {"receptionist_retriever_history":[action],"last_action_outcome": [action] }

    def run_cleaner(self,state):
        action=self.cleaner.invoke(state)
        if "error" in action.content:
            pr="Scraper: I got an error, going back to the arxiv_retriever"
            print(pr)
            return {"last_action_outcome": [action], "should_I_clean":True}
        else:
            pr="Scraper: I got the following paper"+action.content
            print(pr)
            return {"title_of_retrieved_paper":action.content, "last_action_outcome": [action], "should_I_clean":False}

    def call_tool(self,state):
        last_message = state["last_action_outcome"][-1]
        tool_call = last_message.tool_calls[0]
        action = ToolInvocation(tool=tool_call["name"],tool_input=tool_call["args"])
        try:
            response = self.tool_executor.invoke(action)
        except Exception as e:
            response = str(e)
        report=ToolMessage("The tool was called", tool_call_id=tool_call["id"])
        response=ToolMessage(response, tool_call_id=tool_call["id"])
        if tool_call["name"] == "get_id_from_url":
            pr="Tool_executor: I am going to execute"+ str(tool_call["name"])+ "with"+ str(tool_call["args"])
            print(pr)
            return {"last_action_outcome": [report],"metadata": response, "should_I_clean":True}
        elif tool_call["name"] == "download_pdf":
            pr="Tool_executor: I am going to execute"+ str(tool_call["name"])+ "with"+ str(tool_call["args"])
            print(pr)
            return {"last_action_outcome": [response]}
    
    def should_continue_receptionist(self,state):
        messages = state["receptionist_retriever_history"]
        last_message = messages[-1]
    # If there is no function call, then we finish
        if  "We are done" in str(last_message.content):
            return "end"
        else:
            return "continue"
    
    def should_continue_retriever(self,state):
        message = state["last_action_outcome"][-1]

    # If there is no function call, then we finish
        if "tool_calls" in message.additional_kwargs:
            return "continue"
    # Otherwise if there is, we continue
        else:
            print("Reporting to receptionist")
            return "receptionist"
    
    def where_next(self,state):
        if state["should_I_clean"]==True:
            return "cleaner"
    # Otherwise if there is, we continue
        else:
            return "retriever"

    def create_workflow(self):
        workflow = StateGraph(ArxivState)
        workflow.set_entry_point("receptionist")
        workflow.add_node("receptionist",self.run_receptionist)
        workflow.add_conditional_edges("receptionist",self.should_continue_receptionist,{"end":END,"continue": "retriever"})
        workflow.add_node("retriever", self.run_retriever)
        workflow.add_conditional_edges("retriever",self.should_continue_retriever,{"continue": "tools","receptionist": "receptionist",})
        workflow.add_node("tools", self.call_tool)
        workflow.add_node("cleaner", self.run_cleaner)
        workflow.add_conditional_edges("tools", self.where_next,{"cleaner": "cleaner","retriever": "retriever",})
        workflow.add_edge("cleaner", "retriever")
        return workflow


class OcrEnchancingWorkflow():
    def __init__(self, enhancer_model=None,  embeder=None):
        if enhancer_model==None:
            self.enhancer_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.enhancer_model = enhancer_model
        if embeder==None:
            self.embeder=OpenAIEmbeddings(model="text-embedding-3-small")
        else:
            self.embeder = embeder

        self.enhancer= ocr_enhancer_prompt_template | self.enhancer_model

    def run_enhancer(self,state):
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        main_text_filename=state["main_text_filename"].content
        main_text_filename=get_filename_without_extension(main_text_filename)
        supporting_text_filename=state["supporting_text_filename"].content
        supporting_text_filename=get_filename_without_extension(supporting_text_filename)
        with open(f"files/markdowns/{supporting_text_filename}.mmd") as f:
            supporting = f.read()

        with open(f"files/markdowns/{main_text_filename}.mmd") as f:
            main = f.read()    

        supporting_splitted_list = text_splitter.split_text(supporting)
        main_splitted_list = text_splitter.split_text(main)
        good_embed=self.embeder.embed_documents(main_splitted_list)
        bad_embed=self.embeder.embed_documents(supporting_splitted_list)
        similarities = util.pytorch_cos_sim(good_embed, bad_embed)
        print("Enhancing started")
        for i in tqdm(range(len(main_splitted_list))):
            main_text_indexed=main_splitted_list[i]
            supporting_text_temp=""
            values, indices = torch.topk(similarities[i], 2)
            for index in indices:
                supporting_text_indexed=supporting_text_temp+supporting_splitted_list[index]
            result=remove_up_to_first_newline(self.enhancer.invoke(input={"good_text":main_text_indexed, "bad_text":supporting_text_indexed}).content)
            main_splitted_list[i]=result
    

        reconstructed_text = ''.join(main_splitted_list)
        with open(f"files/markdowns/{main_text_filename}_enhanced.mmd", 'w') as file:
            file.write(reconstructed_text)
        return {"report":HumanMessage(content="Done!")}
    def create_workflow(self):
        workflow = StateGraph(OcrEnchancerState)
        workflow.set_entry_point("enhancer")
        workflow.add_node("enhancer",self.run_enhancer)
        workflow.add_edge("enhancer", END)
        return workflow



class ProofRemovingWorkflow:
    def __init__(self, remover_model=None, stamper_model=None):
        if remover_model==None:
            self.remover_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.remover_model = remover_model
        if stamper_model==None:
            self.stamper_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.stamper_model = stamper_model
        self.remover = proof_remover_prompt_template | self.remover_model
        self.stamper= proof_stamper_prompt_template | self.stamper_model
    def run_stamper(self, state):
        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        main_text_filename=state["main_text_filename"].content
        main_text_filename=get_filename_without_extension(main_text_filename)
        with open(f"files/markdowns/{ main_text_filename}.mmd", "r",encoding='utf-8') as f:
            text = f.read()    

        listed_text = text_splitter.split_text(text)
        print("Stamping phase is initiated.")
        for i in tqdm(range(len(listed_text))):
            is_proof=self.stamper.invoke(input={"text":listed_text[i]})
            if is_proof.content=="Yes":
                listed_text[i+1]="(PROOF CONTINOUS FROM PREVIOUS PAGE)" + listed_text[i+1]

        return {"file":listed_text, "main_text_filename":main_text_filename}
    
    def run_remover(self, state):
        listed_text=state["file"]
        main_text_filename=state["main_text_filename"]
        print("Proof removal in progress")    
        finalwithoutproofs=""
        for i in tqdm(range(len(listed_text))):
            result=self.remover.invoke(input={"text":listed_text[i]}).content
            finalwithoutproofs = finalwithoutproofs + result
        with open(f"files/markdowns/{main_text_filename}_without_proofs.mmd","w",encoding='utf-8') as f:
            f.write(finalwithoutproofs)
        report= "The proofs were remove and the resulted file is named " + main_text_filename + "_without_proofs"
        print(report)
        return {"report":HumanMessage(content=report)}
    
    def create_workflow(self):
        workflow = StateGraph(ProofRemoverState)
        workflow.set_entry_point("proof_stamper")
        workflow.add_node("proof_remover",self.run_remover)
        workflow.add_node("proof_stamper",self.run_stamper)
        workflow.add_edge("proof_stamper", "proof_remover")
        workflow.add_edge("proof_remover", END)
        return workflow


class TranslationWorkflow:
    def __init__(self, translator_model=None):
        if translator_model==None:
            self.translator_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.translator_model = translator_model
        self.translator = translator_prompt_template | self.translator_model
        
    def run_translator(self, state):
        keywords_and_summary_filename = state["keywords_and_summary_filename"].content
        target_language = state["target_language"].content
        main_text_filename = state["main_text_filename"].content
        main_text_filename=get_filename_without_extension(main_text_filename)
        keywords_and_summary_filename=get_filename_without_extension(keywords_and_summary_filename)

        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        with open(f"files/markdowns/{main_text_filename}.mmd","r", encoding='utf-8') as f:
                text = f.read()
        try:
            with open(f"files/markdowns/{keywords_and_summary_filename}.mmd","r", encoding='utf-8') as f:
                keyword_and_summary = f.read()
        except FileNotFoundError:
            print("File not found: The keyword_and_summary file does not exist. Assuming keyword_and_summary is blank.")
            keyword_and_summary = " "

        if "_without_proofs" in main_text_filename:
            main_text_filename = main_text_filename.replace("_without_proofs", "")

        listed_text = text_splitter.split_text(text)
        translation = ""

        print(f"Translation of {main_text_filename} in progress")
        
        for i in tqdm(range(len(listed_text))):
            translation = translation + self.translator.invoke({"language": target_language, "keyword_and_summary": keyword_and_summary, "page": listed_text[i]}).content

        with open(f"files/markdowns/{main_text_filename}_{target_language}.mmd", "w", encoding="utf-8") as f:
            f.write(translation)

        return {"report": HumanMessage(content="Translation completed")}

    def create_workflow(self):
        workflow = StateGraph(TranslatorState)
        workflow.set_entry_point("translator")
        workflow.add_node("translator", self.run_translator)
        workflow.add_edge("translator", END)
        return workflow


class KeywordAndSummaryWorkflow:
    def __init__(self, keyword_and_summary_maker_model=None):
        if keyword_and_summary_maker_model==None:
            self.keyword_and_summary_maker_model=ChatNVIDIA(model="meta/llama3-70b-instruct")
        else:
            self.keyword_and_summary_maker_model = keyword_and_summary_maker_model
        self.keyword_and_summary_maker= keyword_and_summary_maker_template | self.keyword_and_summary_maker_model
    
    def run_keyword_and_summary_maker(self, state):
        text_name = state["main_text_filename"].content
        text_name=get_filename_without_extension(text_name)
        with open(f"files/markdowns/{text_name}.mmd", 'r', encoding='utf-8') as f:
            text = f.read()

        text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
        text = text_splitter.split_text(text)
        keyword_and_summary = ""
        print("keyword_and_summary in progress")
        for i in tqdm(range(len(text))):
            keyword_and_summary = self.keyword_and_summary_maker.invoke({"text": keyword_and_summary, "page": text[i]}).content

        output_filename = f"files/markdowns/{text_name}_keyword_and_summary.mmd"
        with open(output_filename, 'w', encoding='utf-8') as file:
            file.write(keyword_and_summary)
        
        report = f"keyword_and_summary completed successfully and the resulted file is named {text_name}_keyword_and_summary"
        print(report)
        return {"report": report}

    def create_workflow(self):
        """
        Create a workflow that executes the keyword and summary extraction.
        """
        workflow = StateGraph(KeywordSummaryState)
        workflow.set_entry_point("summarizer")
        workflow.add_node("summarizer", self.run_keyword_and_summary_maker)
        workflow.add_edge("summarizer", END)
        return workflow
    