from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langchain_core.messages import  BaseMessage, ToolMessage, AIMessage
from langgraph.prebuilt import ToolInvocation
from langgraph.prebuilt.tool_executor import ToolExecutor 
from typing import TypedDict, Annotated
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from prompts import *
from simple_tools import *
from workflows_as_tools import *
import streamlit as st
from dotenv import load_dotenv


class MetaState(TypedDict):
    manager_history: Annotated[list[BaseMessage], operator.add]
    folder_structure: str

class MetaWorkflow():
    def __init__(self,model):
        
        self.supervisor_model=model
        self.tools=create_tools()
        self.supervisor=supervisor_prompt_template | self.supervisor_model.bind_tools(self.tools)
        self.tool_executor=ToolExecutor(self.tools)
        self.folder_structure = get_folder_structure()

    def supervisor_run(self,state):
        action = self.supervisor.invoke(state)
        print(action.content)
        return({"manager_history": [action]})
    def call_tool(self,state):
        last_message = state["manager_history"][-1]
        tool_call = last_message.tool_calls[0]
        action = ToolInvocation(tool=tool_call["name"],tool_input=tool_call["args"])
        try:
            response = self.tool_executor.invoke(action)
        except Exception as e:
            response = str(e)
        print(response)
        response=ToolMessage(content=response, tool_call_id=tool_call["id"])
        return({"manager_history": [response]})
    def user_run(self,state):
        action=HumanMessage(content=input("Enter your answer/querry: "))
        print(action)
        return({"manager_history": [action]})
    def where_next_supervisor(self,state):
        if "tool_calls" in state["manager_history"][-1].additional_kwargs:
            return("tools")
        else:
            return("user")
    def where_next_user(self,state):
        if "exit" in state["manager_history"][-1].content:
            return("end")
        else:
            return("supervisor")

    def create_workflow(self):
        workflow = StateGraph(MetaState)
        workflow.set_entry_point("user")
        workflow.add_node("supervisor",self.supervisor_run)
        workflow.add_node("tools", self.call_tool)
        workflow.add_node("user", self.user_run)
        workflow.add_edge("tools","supervisor")
        workflow.add_conditional_edges("supervisor",self.where_next_supervisor,{"tools":"tools","user":"user"})
        workflow.add_conditional_edges("user",self.where_next_user,{"end":END,"supervisor":"supervisor"})
        workflow.add_edge("tools","user")
        return workflow


model=ChatOpenAI(model="gpt-4o",temperature=0)
workflow=MetaWorkflow(model)
app=workflow.create_workflow()
app=app.compile()
print("How can I assist you today?")
app.invoke({"manager_history": [AIMessage(content="How can I help you today")],"folder_structure": get_folder_structure()})