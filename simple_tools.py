import urllib, os, subprocess, pathlib, pymupdf4llm
from langchain_core.tools import tool



@tool
def get_id_from_url(url:str) -> str:
    """This is a search tool inside arxiv. 
    it takes a query in the form of a urland returns  metadata like the id of the paper that answers better to the query."""
    try:
        with urllib.request.urlopen(url) as response:
            return response.read()
    except urllib.error.HTTPError as e:
        # Handle HTTP errors (e.g., 404, 500)
        print(f'HTTPError: {e.code} - {e.reason}')
        return f'Error: {e.code} - {e.reason}'
    except urllib.error.URLError as e:
        # Handle URL errors (e.g., network issues, invalid URL)
        print(f'URLError: {e.reason}')
        return f'Error: {e.reason}'
    except Exception as e:
        # Handle any other exceptions
        print(f'Unexpected error: {e.reason}')
        return f'Error: {e.reason}'

@tool
def download_pdf(id:str, title:str)->str:
    """This tool takes an id of an arXiv paper and a short title, downloads the pdf file
    corresponding to the id and saves it."""
    
    url = f"https://arxiv.org/pdf/{id}.pdf"
    save_path = os.path.join(r"files\pdfs", f"{title}.pdf")
    # Send a GET request to the URL
    response = urllib.request.urlopen(url)

     # Check if the request was successful
    if response.status == 200:
        # Open a file in write-binary mode and write the response content to it
        with open(save_path, 'wb') as file:
            file.write(response.read())
        a=f"PDF has been downloaded successfully and saved to {save_path}"
    else:
        a=f"Error. Failed to download PDF. Status code: {response.status}"
    return a


@tool
def pdf_to_markdown(pdf_name:str)-> str:
    """This method takes as input the name of a pdf it turns it to markdown"""
    pdf_name=get_filename_without_extension(pdf_name)
    mupdf_name="mu_"+pdf_name
    pdf_path = os.path.join(r"files\pdfs", f"{pdf_name}.pdf")
    print("Processing the PDF with mupdf...")
    mupdf_path = os.path.join(r"files\markdowns", f"{mupdf_name}.mmd")
    print("Processing the PDF with nugat...")
    md_text = pymupdf4llm.to_markdown(pdf_path)
    pathlib.Path(mupdf_path).write_bytes(md_text.encode())
    # Define the command as a list of arguments
    command = ["nougat", pdf_path, "--no-skipping", "-o", "files\\markdowns", "-m", "0.1.0-base"]
    # Execute the command
    try:
        subprocess.check_call(command)
        response="File" + pdf_name + "converted successfully"
    except subprocess.CalledProcessError as e:
        response="Error occurred while converting the file"+ pdf_name +":" +str(e)
    print(response)
    return response



def remove_up_to_first_newline(text):
    # Split the text at the first newline character
    parts = text.split('\n', 1)
    # If there's no newline, return an empty string
    if len(parts) == 1:
        return ''
    # Return the part after the first newline
    return parts[1]

def get_folder_structure():
    current_dir = os.getcwd()
    target_folders = ["files/pdfs", "files/markdowns"]
    folder_structure = {}
    for target_folder in target_folders:
        folder_path = os.path.join(current_dir, target_folder)
        if os.path.exists(folder_path):
            files = [os.path.splitext(file)[0] for file in os.listdir(folder_path)]
            folder_structure[target_folder] = {
                "files": files
            }
    return folder_structure

def get_filename_without_extension(file_path):
    # Extract the base name from the file path
    base_name = os.path.basename(file_path)
    # Split the base name and extract the first part before any dot
    file_name = os.path.splitext(base_name)[0]
    return file_name

def list_files(directory_path):
    try:
        # List all files in the given directory
        files = [os.path.splitext(f)[0] for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
        return files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []