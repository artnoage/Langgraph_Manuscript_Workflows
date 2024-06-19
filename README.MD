# Academic Breaker

## Overview

This project leverages large language model (LLM) agents within LangGraph to develop workflows for processing academic manuscripts. Users can execute these workflows individually through the provided Jupyter notebook (currently the more reliable option) or orchestrate the entire process via a robust LLM accessed through a Streamlit interface. The ultimate aim is to create an app that simplifies some of the more tedious aspects of academic research for researchers. For instance, a tangible goal would be to summarize the state of the art and the historical development of a chosen topic, providing a comprehensive survey. A longer-term goal will be to add the ability for the workflow to explain mathematical proofs and even categorize them.

## Prerequisites

Before you begin, ensure you have the following installed:

- Git
- Python 3.8 or later
- pip (Python package installer)

## Current Workflows

- **Arxiv Paper Retrieval**: Given a list of keywords, this workflow retrieves the most relevant papers from arXiv.
  
- **PDF to Text**: Utilizes Nougat from Meta for OCR conversion of papers. This method, while effective, requires significant computational resources.

- **OCR Enhancer**: Since Nougat sometimes distorts citation formats, this workflow uses MuPDF to generate a secondary text file. Although MuPDF's quality is lower (especially for mathematical content), it corrects citation formats. By comparing the two text files, the workflow merges the best aspects of both.

- **Proof Remover**: Aiming to clean texts by removing proofs before summarization, this workflow has been the least successful. Suggestions for improvement are welcome.

- **Keyword Extraction and Topic Summarization**: This workflow extracts keywords and summarizes the content of an article, page by page. There is potential for improvement by performing multiple passes over the paper.

- **Citation_Extraction**: This workflow extracts citations from a  paper. You can provide a context file (like summary) and a type of citations (Full citations, most important, etc etc)

- **Context-Based Translation**: This workflow has been particularly successful. It uses the summarization from the previous step to translate the article into different languages. Context-based translation ensures that the terminology is appropriate for the specific academic community.


## Extra features

- **Folder awareness**: The manager is aware of the conents of the two folders (pdfs/markdowns) and can guide you to open the right files. Also minimizing the possibility of using the tools wrong

- **Take A Peak**: It looks inside a file for some quick information. Good for getting citations out of a file to push to the ArXiv retrieval

## Upcoming Features

- **Survey Creation**: The LLM will identify the most relevant citations in a paper, retrieve related papers from arXiv or other sources, process each paper using the current workflows, and compile a comprehensive survey. This feature aims to streamline the creation of complex surveys.

- **Proof Explainer**: This ambitious feature intends to analyze mathematical texts, generate context, and explain proofs. A secondary goal is to categorize proofs with similar arguments, enhancing understanding and learning.

This project continues to evolve, aiming to significantly aid academic researchers by automating and improving various manuscript processing tasks.

## Installation

### Clone the Repository
```sh
git clone https://github.com/artnoage/Langgraph_Manuscript_Workflows.git
cd Langgraph_Manuscript_Workflows
```

### Creating the Virtual Environment

#### Using Python `venv`
```sh
python -m venv Langgraph_Manuscript_Workflows
```

#### Using Conda
```sh
conda create -n Langgraph_Manuscript_Workflows python=3.8
```

### Activating the Virtual Environment

#### Python `venv`
- **Windows**
    ```sh
    .\Langgraph_Manuscript_Workflows\Scripts\activate
    ```
- **macOS/Linux**
    ```sh
    source Langgraph_Manuscript_Workflows/bin/activate
    ```

#### Conda (all platforms)
```sh
conda activate Langgraph_Manuscript_Workflows
```

### Installing the Required Dependencies

#### Python `venv`
- **Windows**
    ```sh
    pip install -r requirements_win.txt
    ```
- **macOS/Linux**
    ```sh
    pip install -r requirements_linux.txt
    ```

#### Conda (all platforms)
```sh
pip install -r requirements_conda.txt
```

### Creating the `.env` File

#### 
```sh
python create_env.py
```

### editing the `.env` File by running the command twice
```sh
python create_env.py
```
### Setting Up API Keys

#### OpenAI API Key
If you don't have an OPENAI_API_KEY, get one from [OpenAI](https://platform.openai.com/account/api-keys) and add it to the `.env` file:
```env
OPENAI_API_KEY="your_key"
```

#### NVIDIA API Key
If you don't have an NVIDIA_API_KEY, get one and some free credits from [NVIDIA](https://build.nvidia.com/explore/discover#llama3-70b) and add it to the `.env` file:
```env
NVIDIA_API_KEY="your_key"
```

## Usage

### Running the Project

You have three options to work with the various mini-workflows in the project:

1. **Jupyter Notebook**: Use `playground.ipynb` to run the workflows independently. This option provides more control over the LLMs used and gives insight into the inner workings of each workflow. Recommended for first-time users.

2. **Terminal**: Run the `metaworkflow.py` file:
   ```sh
   python metaworkflow.py
   ```

3. **Streamlit UI**: Run the Streamlit app:
   ```sh
   streamlit run streamlit_app.py
   ```

### Warning

**Streamlit and LLM APIs can have difficulty canceling tool calls while they are running.** For this reason, it is recommended to use small files for testing. Additionally, the tool printouts are not currently forwarded to the Streamlit interface.

## Contributing

Contributions are welcome! You can contribute significantly without any coding knowledge by modifying the system prompts in the `prompts.py` file until the corresponding workflow behaves better. This process is known as "prompt engineering."

If you have experience with Streamlit, you can suggest ways to stream the printouts from the tools to the Streamlit interface.

You can also create different workflows that you think are relevant to manipulating academic manuscripts.

## Acknowledgments

- Slide template: Srikar Sharma Sadhu (https://tinyurl.com/3n2cfpda)
- OCR: Nougat from Meta (https://github.com/facebookresearch/nougat)
- Assistance from ChatGPT and Claude Opus

Special thanks to:
- My girlfriend for her patience during the preparation of this project for the NVIDIA competition.
- My friend Anna for participating in the promotional video.
- Nikos Mouratidis and Kostas Kostopoulos for beta testing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.