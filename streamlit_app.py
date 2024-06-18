from simple_workflows import *
from simple_tools import *
from workflows_as_tools import *
import streamlit as st
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.messages import ToolMessage
from simple_tools import *
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import os
import time
from dotenv import load_dotenv
load_dotenv()
def invoke(state,container):   
    
    supervisor_model=ChatOpenAI(model="gpt-4o",temperature=0)
    tools=create_tools()
    supervisor=supervisor_prompt_template | supervisor_model.bind_tools(tools)
    tool_executor=ToolExecutor(tools)
    folder_structure = get_folder_structure()
    while True:
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
                        try:
                            response = tool_executor.invoke(Invocation)
                        except Exception as e:
                            response = str(e)                   
                        st.write(response)
                        response=ToolMessage(content=response, tool_call_id=tool_call["id"])
                        st.session_state.chat_history.append(response)
                        st.session_state.messages.append({"role": "tool", "content": response.content})
        if "tool_calls" not in action.additional_kwargs:
            with container:
                with st.chat_message(message["role"]):
                    st.write(message["content"])       
            st.session_state.chat_history.append(action)
            st.session_state.messages.append({"role": "assistant", "content": action.content})
            break
    return 



def main():
    st.set_page_config(page_title="Chat with bot that broke academia! HERE HERE", layout="wide")
     # Streamlit page configuration
    st.markdown("<h1 style='text-align: center;margin-left: -550px;'>Chat with the Bot that broke Academia!!</h1>", unsafe_allow_html=True)
    ready = True
    global st_file
    st_file=None
    openai_api_key = os.getenv('OPENAI_API_KEY')
    nvidia_api_key = os.getenv('NVIDIA_API_KEY')
    
    if not openai_api_key:
        st.warning("Missing OPENAI_API_KEY")
        ready = False
    if not nvidia_api_key:
        st.warning("Missing Nvidia_API_KEY")
        ready = False

   
    pdfs = "files\\pdfs"
    mmd = "files\\markdowns"

    
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "I hope you had your morning coffee! Are you ready to get started?"}]
    if 'awaiting_response' not in st.session_state:
        st.session_state.awaiting_response = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if "disable_input" not in st.session_state:
        st.session_state.disable_input = False
        
    # Create a sidebar widget to display the folder structure
    left_sidebar, main_content, right_sidebar = st.columns([0.15,0.45, 0.40],gap="large")
    
# Left sidebar
    with left_sidebar:
        left_container = st.container(height=600)
        with left_container:
            selected_type = st.selectbox("Select a type", ["PDF", "Markdown"], index=None)
            pdf_files = list_files(pdfs)
            markdown_files = list_files(mmd)
            if selected_type == "PDF":
                selected_file = st.selectbox("Select a file", pdf_files, index=None)
            else:
                selected_file = st.selectbox("Select a file", markdown_files, index=None)            
            
            if selected_file:
                if selected_type=="PDF":
                    st_file = os.path.join(pdfs, selected_file+".pdf")
                else:
                    st_file = os.path.join(mmd, selected_file+".mmd")

            uploaded_file = st.file_uploader("Import Manually", type=["pdf", "mmd"])
            # Handle the uploaded file
            if uploaded_file is not None:
                # Save the uploaded file to the specified directory
                _, file_extension = os.path.splitext(uploaded_file.name)
                if file_extension == ".pdf":
                    save_path = os.path.join(pdfs, uploaded_file.name)
                else:
                    save_path = os.path.join(mmd, uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"File '{uploaded_file.name}' uploaded successfully!")
        st.image("files/images/robot2.png", use_column_width=True)    


            
    with right_sidebar:
        right_container = st.container(height=1000)
        with right_container:
            if not openai_api_key:
                openai_api_key = "You dont have an OPENAI_API_KEY, get one from here: https://platform.openai.com/account/api-keys, and put it in the .env file"
            if not nvidia_api_key:
                nvidia_api_key= "You dont have an Nvidia_API_KEY, get one from here: https://org.ngc.nvidia.com/setup/api-key, and put it in the .env file"
            
            if not st_file:
                with open("README.MD", "r") as file:
                    markdown_content = file.read()
                st.markdown(markdown_content)
            elif st_file and selected_type=="PDF":
                pdf_viewer(st_file)                                                        
            elif st_file and selected_type=="Markdown":
                with open(st_file, "r") as file:
                    markdown_content = file.read()
                st.markdown(markdown_content)
           
                
    

# Main content
    if ready:
        with main_content:
            # Chat history container with scrollbar
            chat_container = st.container(height=1000)
            with chat_container:
                for message in st.session_state.messages:
                    if message["role"] == "tool":
                        with st.chat_message(message["role"],avatar=":material/build:"):
                            st.write(message["content"])    
                    else:
                        with st.chat_message(message["role"]):
                            st.write(message["content"])                    
            

            # Display the input widget with the disabled condition
            prompt = st.chat_input("Enter your querry", disabled=st.session_state.disable_input)
            if st.session_state.awaiting_response is False:
                if prompt:
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    st.session_state.chat_history.append(HumanMessage(content=prompt))
                    st.session_state.awaiting_response = True
                    with chat_container:
                        with st.chat_message("user"):
                            st.write(prompt)

            # Generate a new response if last message is not from assistant
            if st.session_state.messages[-1]["role"] == "user" and st.session_state.awaiting_response:
                st.session_state.disable_input = True
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Thinking..."):
                            invoke(st.session_state,chat_container)
                            st.session_state.awaiting_response = False
                            st.session_state.disable_input = False
                st.rerun()
    else:
        st.stop()
        
    

    
    
    

if __name__ == "__main__":
    main()
