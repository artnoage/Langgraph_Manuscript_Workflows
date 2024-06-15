from dotenv import load_dotenv
from langgraph.graph import END, StateGraph
from langchain_core.messages import  BaseMessage, ToolMessage
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


supervisor_model=ChatOpenAI(model="gpt-4o",temperature=0)
tools=create_tools()
supervisor=supervisor_prompt_template | supervisor_model.bind_tools(tools)
tool_executor=ToolExecutor(tools)
folder_structure = get_folder_structure()
exit=False
while not exit:
    workflow_state = {"manager_history": state.chat_history, "folder_structure": folder_structure}
    action = supervisor.invoke(workflow_state)
    message=st.session_state.messages[-1]
    if "tool_calls" in action.additional_kwargs:
        with container:
            with st.chat_message(message["role"],avatar=":material/build:"):
                    st.write("I am currently using the following tool: "+ action.tool_calls[-1]["name"])
                    st.write("Please be patient, some of the tools take time. Check your terminal for progress.")
                    st.session_state.chat_history.append(action)
                    st.session_state.messages.append({"role": "assistant", "content": "I am currently using the following tool: " + action.tool_calls[-1]["name"]})
                    tool_call = action.tool_calls[-1]
                    st.write(tool_call["name"],tool_call["args"])
                    Invocation=ToolInvocation(tool=tool_call["name"], tool_input=tool_call["args"])
                    st.write(Invocation)
                    response = tool_executor.invoke(Invocation)
                    
                    st.write(response)
                    response=ToolMessage(response, tool_call_id=tool_call["id"])
                    st.session_state.chat_history.append(response)
                    st.session_state.messages.append({"role": "tool", "content": response.content})
    if "tool_calls" not in action.additional_kwargs:
        with container:
            with st.chat_message(message["role"]):
                st.write(message["content"])       
        st.session_state.chat_history.append(action)
        st.session_state.messages.append({"role": "assistant", "content": action.content})
        break

