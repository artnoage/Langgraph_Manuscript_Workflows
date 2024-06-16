# Academic Manuscript Processing

## Overview

This project leverages large language model (LLM) agents within LangGraph to develop workflows for processing academic manuscripts. Users can execute these workflows individually through the provided Jupyter notebook (currently the more reliable option) or orchestrate the entire process via a robust LLM accessed through a Streamlit interface. The ultimate aim is to create an app that simplifies some of the more tedious aspects of academic research for researchers. For instance, a tangible goal would be to summarize the state of the art and the historical development of a chosen topic, providing a comprehensive survey.

## Current Workflows

- **Arxiv Paper Retrieval**: Given a list of keywords, this workflow retrieves the most relevant papers from arXiv.
  
- **PDF to Text**: Utilizes Nougat from Meta for OCR conversion of papers. This method, while effective, requires significant computational resources.

- **OCR Enhancer**: Since Nougat sometimes distorts citation formats, this workflow uses MuPDF to generate a secondary text file. Although MuPDF's quality is lower (especially for mathematical content), it corrects citation formats. By comparing the two text files, the workflow merges the best aspects of both.

- **Proof Remover**: Aiming to clean texts by removing proofs before summarization, this workflow has been the least successful. Suggestions for improvement are welcome.

- **Keyword Extraction and Topic Summarization**: This workflow extracts keywords and summarizes the content of an article, page by page. There is potential for improvement by performing multiple passes over the paper.

- **Context-Based Translation**: This workflow has been particularly successful. It uses the summarization from the previous step to translate the article into different languages. Context-based translation ensures that the terminology is appropriate for the specific academic community.

## Upcoming Features

- **Survey Creation**: The LLM will identify the most relevant citations in a paper, retrieve related papers from arXiv or other sources, process each paper using the current workflows, and compile a comprehensive survey. This feature aims to streamline the creation of complex surveys.

- **Proof Explainer**: This ambitious feature intends to analyze mathematical texts, generate context, and explain proofs. A secondary goal is to categorize proofs with similar arguments, enhancing understanding and learning.

This project continues to evolve, aiming to significantly aid academic researchers by automating and improving various manuscript processing tasks.


# Installation:

### Clone the repository:
1. 
    ```sh
    git clone https://github.com/artnoage/Langgraph_Manuscript_Workflows.git
    ```

### Navigate to the project's direcotry:
2. 
    ```sh
    cd Langgraph_Manuscript_Workflows
    ```

### Creating the virtual environment with python venv:

3.  
    ```sh
    python -m venv Langgraph_Manuscript_Workflows 
    ```

### Creating the virtual environment with conda virtual environment:
    
 4. 
    ```sh
    conda create -n Langgraph_Manuscript_Workflows python=3.8
    ```

### Activating the virtual environment python when you are in windows:

4. 
    ```sh
    source Langgraph_Manuscript_Workflows/scripts/activate
    ```
### Activating the virtual enviroment python linux (probably mac too):

6.
```sh
source Langgraph_Manuscript_Workflows/bin/activate
```   
### Activating the virtual environment conda regardless of platform:

7
```sh
conda activate Langgraph_Manuscript_Workflows
```
### Installing the required dependencies python windows:

8.  
```sh
pip install -r requirements_win.txt
```  
### Intsalling the required dependencies python linux (probably mac too):
9. 
```sh
pip install -r requirements_linux.txt
```
### Intalling the required dependencies conda:
10. 
```sh
pip install -r requirements_conda.txt
```

### Creating the .env file:
11. 
```sh
touch .env
```
### Gettomg an openAI api key:
12.
```sh
If you dont have an OPENAI_API_KEY, get one from here: https://platform.openai.com/account/api-keys, and put it in the .env file like this: OPENAI_API_KEY = "your key"
```
### Getting a Nvidia Api key:
13. 
```sh
If you dont have an NVIDIA_API_KEY, get one from here: https://org.ngc.nvidia.com/setup, get your free credits here: https://build.nvidia.com/explore/discover. Put the key in the .env file like this: NVIDIA_API_KEY = "your key"
 ```


# Running the project:
You have three options to work with the various mini-workflows in the project. 
One is through the jupyter notebook called playground.ipnyb and the other through 
the fronent that uses streamlit for the user interface. 

In the playground you can find details for what is happening under the hood for each workflow. 
Evenmore you have a better control of which llms you are using. I suggest this for the first time.

## First method by trying the workflows independently:
```sh   
Use google colab or VScode with Jupyter extension.
``` 
## Second method by using terminal:
```sh   
python metaworkflow.py
``` 
## Third method by using streamlit:
```sh
streamlit run streamlit_app.py 
```
 ### WARNING
 **Streamlit is a bit difficult with calcellation signals. Same holds for LLM Apis. It will be difficult or impossible
 to cancel a tool call when it is running. For this reason, I recommend using small files for testing. Also the tools have printouts which I didnt manage to forward somehow in streamlit. When I get levels in using it, I will try to improve.**

## Contributing

Contributions are welcome! One can contribute significantly without any knowledge of coding. You
just go to the prompts file and change the system prompts until the corresponding workflow behaves better 
than you find it. The term prompt engineering is annoying but when it comes to the currest state of LLMs
it makes sense.
If you are experienced with streamlit, maybe you can make me suggestions how I can stream the printouts I get from
my tools. I dont have experience in front end and this seems ultra hard to me.

You could also create different workflows that you think you are relevant to manipulating academinc manuscripts.

## Shoulders of Giants

I got the slide template by: Srikar Sharma Sadhu (https://tinyurl.com/3n2cfpda)
For ocr I used nougat : https://github.com/facebookresearch/nougat
I used alot of ChatGPT and Claude Opus 

I would like also to thank:
My girfried for being patient with me the whole month preparing this project for Nvidia competition.
My friend Anna for participating in the promotional video. 
Nikos Mouratidis and Kostas Kostopoulos for beta testing. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
