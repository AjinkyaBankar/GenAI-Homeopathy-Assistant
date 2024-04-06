# Load the text 
from langchain_community.document_loaders import TextLoader
loader = TextLoader("./encyclopedia_of_homeopathy.txt")
loader.load()
pages = loader.load()

len(pages)
print(pages[0].page_content)


# Split the text
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 20,
    separators=["\n\n", "(?<=\. )", " ", "\n", "," ""] #Default separator priority is altered
)
splits = text_splitter.split_documents(pages)


# Load the embeddings model
from langchain.embeddings import HuggingFaceEmbeddings
hf_embed = HuggingFaceEmbeddings(model_name='BAAI/bge-large-en-v1.5')


import os
local_dir_out = os.path.join(os.getcwd(), "local_dir")
if not os.path.exists(local_dir_out):
    os.makedirs(local_dir_out)


# Index the vector database by embedding then inserting document chunks
vector_db_path = os.path.join(local_dir_out,"Homeopathy")

from langchain.vectorstores import Chroma
vectordb = Chroma.from_documents(
    collection_name="homeopathy_book",
    documents=splits,
    embedding=hf_embed,
    persist_directory=vector_db_path
)

vectordb.persist()


# Prompt Engineering
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM, BitsAndBytesConfig, TextIteratorStreamer
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory

model_name = "MBZUAI/LaMini-Flan-T5-783M"
instruct_pipeline = pipeline(model=model_name, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", max_length=512, max_new_tokens=64, top_p=0.95, top_k=50)
hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)

def build_qa_chain():
  torch.cuda.empty_cache()

  # Defining our prompt content.
  # langchain will load our similar documents as {context}
  template = """Instruct: You are an AI assistant for answering questions about the provided context. You are given the following extracted parts of a long document and a question. Provide a detailed answer. If you don't know the answer, just say "Hmm, I'm not sure." Don't try to make up an answer.
  #########################
  {context}
  #########################
  Question: {human_input}

  Output:\n"""
  prompt = PromptTemplate(input_variables=['context', 'human_input'], template=template)

  # Set verbose=True to see the full prompt:
  print("loading chain, this can take some time...")
  return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt, verbose=True)


# Building the chain will load model and can take several minutes depending on the model size
qa_chain = build_qa_chain()

def displayHTML(html):
    """Display HTML in Jupyter notebook."""
    from IPython.display import HTML
    display(HTML(html))

class ChatBot():
  def __init__(self, db):
    self.reset_context()
    self.db = db

  def reset_context(self):
    self.sources = []
    self.discussion = []
    # Building the chain will load Dolly and can take some time depending on the model size and your GPU
    self.qa_chain = build_qa_chain()
    displayHTML("<h1>Hi! I'm a Homeopathy chatbot. How Can I help you today?</h1>")

  def get_similar_docs(self, question, similar_doc_count):
    return self.db.similarity_search(question, k=similar_doc_count)

  def chat(self, question):
    self.discussion.append(question)
    similar_docs = self.get_similar_docs(question, similar_doc_count=2)

    result = self.qa_chain({"input_documents": similar_docs, "human_input": question})
    # Cleanup the answer for better display:
    answer = result['output_text'].capitalize()
    result_html = f"<p><blockquote style=\"font-size:24\">{question}</blockquote></p>"
    result_html += f"<p><blockquote style=\"font-size:18px\">{answer}</blockquote></p>"
    result_html += "<p><hr/></p>"
    for d in result["input_documents"]:
      source_id = d.metadata["source"]
      self.sources.append(source_id)
      result_html += f"<p><blockquote>{d.page_content}<br/>(Source: <a href=\"/{source_id}\">{source_id}</a>)</blockquote></p>"
    displayHTML(result_html)

chat_bot = ChatBot(vectordb)


# Ask the queries
chat_bot.chat("which homeopathy remedy is useful for persons who tend to be sensitive and emotionally dependent on others?")

chat_bot.chat("which type of patients Belladonna is suitable for?")

chat_bot.chat("patient has sore throat, running nose and fear of severe disease. which remedy is useful to treat?")