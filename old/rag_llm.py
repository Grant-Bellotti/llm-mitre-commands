import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline
)
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

# ================================
# 1. Load and split your document
# ================================
file_path = "example.txt"   # <-- replace with your file
loader = TextLoader(file_path)
documents = loader.load()
print(f"Loaded {len(documents)} document(s).")

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = splitter.split_documents(documents)
print(f"Split into {len(docs)} chunks.")

# ==================================
# 2. Create embeddings + vectorstore
# ==================================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(docs, embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ==================================
# 3. Load quantized local Llama model
# ==================================
model_id = "meta-llama/Llama-3.1-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

# Wrap in a transformers pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.2,
    top_p=0.9,
    do_sample=True
)

# Wrap pipeline for LangChain
llm = HuggingFacePipeline(pipeline=pipe)

# ==================================
# 4. Build RetrievalQA chain
# ==================================
qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# ==================================
# 5. Ask a question
# ==================================
query = "What are the main points discussed in the document?"
answer = qa.run(query)

print("\n====================")
print("Q:", query)
print("A:", answer)
print("====================")
