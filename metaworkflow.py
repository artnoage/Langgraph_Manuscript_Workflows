from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langchain_core.messages import  BaseMessage, ToolMessage
from langgraph.prebuilt import ToolInvocation
from langgraph.prebuilt.tool_executor import ToolExecutor 
from typing import TypedDict, Annotated
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
import operator
from prompts import *
from simple_tools import *
from workflows_as_tools import *
import streamlit as st
from io import StringIO
import sys
from dotenv import load_dotenv

load_dotenv()


class MetaState(TypedDict):
    manager_history : Annotated[list[BaseMessage], operator.add]

    
    
    
def meta_workflow(supervisor_model=llm1):
    tools =  [translate_file]
    supervisor=supervisor_prompt_template | supervisor_model.bind_tools(tools)
    tool_executor=ToolExecutor(tools)
    def run_supervisor(state):
        action = supervisor.invoke(state)
        return {"manager_history":[action]}
    
    def user(state):
        xuser_input = state["manager_history"][-1].content
        return {"manager_history":[HumanMessage(content=xuser_input)]}
    
    def call_tool(state):
        last_message = state["manager_history"][-1]
        tool_call = last_message.tool_calls[0]
        action = ToolInvocation(tool=tool_call["name"],tool_input=tool_call["args"])
        response = tool_executor.invoke(action)
        return {"manager_history": [response] }

    def where_next_human(state):
        print(state)
        last_message = state["manager_history"][-1].content
        if last_message=="exit":
            return "end"
        else:
            return "supervisor"
    def where_next_supervisor(state):
        last_message = state["manager_history"][-1]
        if "tool_calls" in last_message.additional_kwargs:
            return "tool"
        else:
            return "user" 

    workflow = StateGraph(MetaState)
    workflow.set_entry_point("supervisor")
    workflow.add_node("user",user)
    workflow.add_node("supervisor", run_supervisor)
    workflow.add_node("tool_executor", call_tool)
    workflow.add_edge("tool_executor", "supervisor")
    
    workflow.add_conditional_edges("user",where_next_human, {"supervisor": "supervisor","end": END})
    workflow.add_conditional_edges("supervisor", where_next_supervisor,{"tool": "tool_executor","user": "user",})
  
    return workflow

