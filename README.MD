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


## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/artnoage/Langgraph_Manuscript_Workflows.git
    ```
2. Navigate to the project directory:
    ```sh
    cd Langgraph_Manuscript_Workflows
    ```
3. Create a virtual environment. Use the first command for python enviroment and the second for conda enviroment:
    ```sh
    python -m venv Langgraph_Manuscript_Workflows or alternatively
    ```
    ```
    conda create -n Langgraph_Manuscript_Workflows python=3.8
    ```
4. Activate the virtual environment. Use the first command for python enviroment and the second for conda enviroment:
    ```sh
    source Langgraph_Manuscript_Workflows/bin/activate
    ```
    ```sh
    conda activate Langgraph_Manuscript_Workflows
    ```
5. Install the required dependencies. Use the first command for python enviroment and the second for conda enviroment:
    ```sh
    pip install -r requirements.txt
    ```  
    ```sh
    pip install -r requirements_conda.txt
    ```
6. Create an enviroment file:
    ```sh
    touch .env
    ```

6. Getting an OpenAi Api key:
    ```sh
    If you dont have an OPENAI_API_KEY, get one from here: https://platform.openai.com/account/api-keys, and put it in the .env file like this: OPENAI_API_KEY = "your key"
    ```
7. Getting a Nvidia Api key:
    ```sh
    If you dont have an NVIDIA_API_KEY, get one from here: https://developer.nvidia.com/nvidia-smi, and put it in the .env file like this: NVIDIA_API_KEY = "your key"
    ```
## Running

You have two options to work with the various mini-workflows in the project. 
One is through the jupyter notebook called playground.ipnyb and the other through 
the fronent that uses streamlit for the user interface. 

In the playground you can find details for what is happening under the hood for each workflow. 
Evenmore you have a better control of which llms you are using. I suggest this for the first time.



## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## Shoulders of Giants

I got the slide template by: Srikar Sharma Sadhu (https://tinyurl.com/3n2cfpda)
For ocr I used nougat : https://github.com/facebookresearch/nougat
I used alot of ChatGPT and Claude Opus 

I would like also to thank:
My girflried for being patient with me the whole month preparing this project for Nvidia competition.
My friend Anna for participating in the promotional video. 
Nikos Mouratidis and Kostas Kostopoulos for beta testing. 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
