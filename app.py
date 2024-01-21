from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.llm_summarization_checker.base import LLMSummarizationCheckerChain
from langchain.prompts import PromptTemplate
import os
import gradio as gr
import shutil
import re
import tempfile
import cache
from pathlib import Path

api_token=os.environ['api']
os.environ["HUGGINFACEHUB_API_TOKEN"]=api_token

temp_dir = "/content/sample_data"

def data_ingestion(file_path):
    if not os.path.exists(file_path):
      raise ValueError(f"File path {file_path} does not exist.")

    path = Path(file_path)
    file_ext = path.suffix

    # file_ext = os.path.splitext(file_path)[-1]
    # if file_ext == ".pdf":
    #     # loader = PyPDFLoader(file_path)
    #     loader = PDFMinerLoader(file_path)
    #     document= loader.load()

    # elif file_ext in {".docx", ".doc"}:
    #     loader = Docx2txtLoader(file_path)
    #     document= loader.load()

    # elif file_ext == ".txt":
    #     loader = TextLoader(file_path)
    #     document= loader.load()

    loader = PDFMinerLoader(file_path)
    document= loader.load()

    length = len(document[0].page_content)

    # Replace CharacterTextSplitter with RecursiveCharacterTextSplitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    split_docs = text_splitter.split_documents(document)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

    llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        model_kwargs={"temperature":1, "max_length":10000},
                        huggingfacehub_api_token=api_token)

    return split_docs

# text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
#     chunk_size=2000, chunk_overlap=0
# )
# split_docs = text_splitter.split_documents(document)

# documents=split_text_into_batches(str(document),400)
# len(documents)
# documents[0]
# #
# from langchain.text_splitter import CharacterTextSplitter
# text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
# documents = text_splitter.split_documents(document)
# Embeddings

# from langchain.chains.question_answering import load_qa_chain

########## CHAIN 1 norm text

def chain1():
    prompt_template = """Write a concise summary of the following:
    {text}
    SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary\n"
        # "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary in English"
        "If the context isn't useful, return the original summary." )

    refine_prompt = PromptTemplate.from_template(refine_template)
    chain1 = load_summarize_chain(
        llm=HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        model_kwargs={"temperature":1, "max_length":10000},
                        huggingfacehub_api_token=api_token),
        chain_type="refine",
        question_prompt=prompt,
        # refine_prompt=refine_prompt,
        return_intermediate_steps=False,
        input_key="input_documents",
        output_key="output_text",
    )
    return chain1

# result = chain({"input_documents":split_docs}, return_only_outputs=True)

########## CHAIN 2 research paper

def chain2():
    prompt_template = """This is a Research Paper,your job is to summarise the text portion without any symbols or special characters, skip the mathematical equations for now:
    {text}
    SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = (
        "Your job is to produce a final summary\n"
        # "We have provided an existing summary up to a certain point: {existing_answer}\n"
        "We have the opportunity to refine the existing summary"
        "(only if needed) with some more context below.\n"
        "------------\n"
        "{text}\n"
        "------------\n"
        "Given the new context, refine the original summary in English"
        "If the context isn't useful, return the original summary." )

    refine_prompt = PromptTemplate.from_template(refine_template)
    chain2 = load_summarize_chain(
        llm = HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        model_kwargs={"temperature":1, "max_length":10000},
                        huggingfacehub_api_token=api_token),
        chain_type = "refine",
        question_prompt = prompt,
        # refine_prompt = refine_prompt,
        return_intermediate_steps=False,
        input_key="input_documents",
        output_key="output_text",
    )
    return chain2

# result = chain({"input_documents":split_docs}, return_only_outputs=True)

########## CHAIN 3 arxiv_paper_1

def chain3():
    prompt_template = """You are being given a markdown document with headers, this is part of a larger arxiv paper. Your job is to write a summary of the document.
        here is the content of the section:
        "{text}"

        SUMMARY:"""
    prompt = PromptTemplate.from_template(prompt_template)

    refine_template = ("""You are presented with a collection of text snippets. Each snippet is a summary of a specific section from an academic paper published on arXiv. Your objective is to synthesize these snippets into a coherent, concise summary of the entire paper.

        DOCUMENT SNIPPETS:
        "{text}"

        INSTRUCTIONS: Craft a concise summary below, capturing the essence of the paper based on the provided snippets.
        It is also important that you highlight the key contributions of the paper, and 3 key takeaways from the paper.
        Lastly you should provide a list of 5 questions that you would ask the author of the paper if you had the chance. Remove all the backslash n (\n)
        SUMMARY:
        """
        )

    refine_prompt = PromptTemplate.from_template(refine_template)
    chain3 = load_summarize_chain(
        llm=HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                        model_kwargs={"temperature":1, "max_length":10000},
                        huggingfacehub_api_token=api_token),
        chain_type="refine",
        question_prompt=prompt,
        # refine_prompt=refine_prompt,
        return_intermediate_steps=False,
        input_key="input_documents",
        output_key="output_text",
    )
    return chain3
# result = chain({"input_documents":split_docs}, return_only_outputs=True)
# chain.run(document)
# print(result["output_text"])

def chain_function(checkbox_values):

    if "Research Paper" in checkbox_values:
        output = chain3()
    elif "Legal Document" in checkbox_values:
        output = chain2()
    elif "Study Material" in checkbox_values:
        output = chain1()
    else:
        output = "Please select a document type to run."
    return output

def result(chain, split_docs):
    summaries = []
    for doc in split_docs:
        result = chain({"input_documents": [doc]})
        # result = chain({"input_documents": [doc]}, return_only_outputs=True)
        summaries.append(result["output_text"])
    text_concat = ""
    for i in summaries:
      text_concat += i
    # output = re.sub(r'\n',"  ","   ",text_concat)
    return text_concat

title = """<p style="font-family:Century Gothic; text-align:center; font-size: 100px">S  I  M  P  L  I  F  Y</p>"""

# description = r"""<p style="font-family: Century Gothic; text-align:center; font-size: 100px">S  I  M  P  L  I  F  Y</p>
# """

# article = r"""
# If PhotoMaker is helpful, please help to ⭐ the <a href='https://github.com/TencentARC/PhotoMaker' target='_blank'>Github Repo</a>. Thanks!
# [![GitHub Stars](https://img.shields.io/github/stars/TencentARC/PhotoMaker?style=social)](https://github.com/TencentARC/PhotoMaker)
# ---
# 📝 **Citation**
# <br>
# If our work is useful for your research, please consider citing:
# ```bibtex
# @article{li2023photomaker,
#   title={PhotoMaker: Customizing Realistic Human Photos via Stacked ID Embedding},
#   author={Li, Zhen and Cao, Mingdeng and Wang, Xintao and Qi, Zhongang and Cheng, Ming-Ming and Shan, Ying},
#   booktitle={arXiv preprint arxiv:2312.04461},
#   year={2023}
# }
# ```
# 📋 **License**
# <br>
# Apache-2.0 LICENSE. Please refer to the [LICENSE file](https://huggingface.co/TencentARC/PhotoMaker/blob/main/LICENSE) for details.
# 📧 **Contact**
# <br>
# If you have any questions, please feel free to reach me out at <b>zhenli1031@gmail.com</b>.
# """

# tips = r"""
# ### Usage tips of PhotoMaker
# 1. Upload more photos of the person to be customized to **improve ID fidelty**. If the input is Asian face(s), maybe consider adding 'asian' before the class word, e.g., `asian woman img`
# 2. When stylizing, does the generated face look too realistic? Adjust the **Style strength** to 30-50, the larger the number, the less ID fidelty, but the stylization ability will be better.
# 3. If you want to generate realistic photos, you could try switching to our other gradio application [PhotoMaker](https://huggingface.co/spaces/TencentARC/PhotoMaker).
# 4. For **faster** speed, reduce the number of generated images and sampling steps. However, please note that reducing the sampling steps may compromise the ID fidelity.
# """

# def process_file(file_obj):
#     destination_path = "/content/sample_data"  # Replace with your desired path
#     shutil.copy(file_obj, destination_path)  # Save file to specified path
#     return os.path.join(destination_path, file_obj)
def process_file(list_file_obj):
    # list_file_path = [x.name for x in list_file_obj if x is not None]
    # file_content = file_obj.data
    # with tempfile.TemporaryFile() as temp_file:
    #     temp_file.write(file_content)
    #     temp_file_path = temp_file.name
    return list_file_obj[0].name

def inference(checkbox_values, uploaded_file):
    file_path = process_file(uploaded_file)
    split_docs = data_ingestion(file_path)
    chain = chain_function(checkbox_values)
    summary = result(chain, split_docs)
    return summary

with gr.Blocks(theme="monochrome") as demo:
    gr.Markdown(title)

    with gr.Row():
        with gr.Column():
            checkbox_values = gr.CheckboxGroup(["Research Paper", "Legal Document", "Study Material"], label="Choose the document type")
            uploaded_file = gr.Files(height=100, file_count="multiple", file_types=["text", ".docx", "pdf"], interactive=True, label="Upload your File.")
            btn = gr.Button("Submit")  # Place the button outside the Row for vertical alignment
        with gr.Column():
            txt = gr.Textbox(
                show_label=False,scale=2,
                # placeholder="Simplify."
            )


    btn.click(
        fn=inference,
        inputs=[checkbox_values, uploaded_file],
        outputs=[txt],
        queue=False
    )
# debug = True
demo.launch(debug = True)

