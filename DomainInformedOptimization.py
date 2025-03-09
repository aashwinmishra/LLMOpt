import os
import numpy as np
import time
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from benchmark_functions import forrester


load_dotenv()
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "books", "forrester.txt")
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )
    loader = TextLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    print("\n--- Creating embeddings ---")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("\n--- Finished creating embeddings ---")

    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        docs, embeddings, persist_directory=persistent_directory)
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=persistent_directory,
                embedding_function=embeddings)


template = """You are an optimization researcher tasked to minimize the value of the Test function
 with 1 input variable x. {background} Current sampled points for 1 variable (x) with their respective Test function values in csv format are:
x, function_value
{values}

Give me a new (x) value that satisfies the following: 
(a) new x is different from all above, 
(b) new x value lies in the range 0 to 1,
(c) has a function value lower than the above function loss values and 
(d) result in a rapid convergence towards the value of x that results in the global minimum of the function. 
Do not write code or any explanation. The output must end with numerical value for (x) only."""

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1})
relevant_docs = retriever.invoke(template)
augmentation = " ".join([doc.page_content for doc in relevant_docs])
prompt_template = ChatPromptTemplate.from_template(template)
vals = f" 0.05, {forrester(0.05)} \n 0.95, {forrester(0.95)} \n 0.5, {forrester(0.5)} \n"
prompt = prompt_template.invoke({"background": augmentation, "values": vals})
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
num_steps = 10

for i in range(num_steps):
    result = model.invoke(prompt)
    x = float(result.content)
    print(f"New Sample: {x}, {forrester(x)}")
    vals += f" {x}, {forrester(x)} \n"
    prompt = prompt_template.invoke({"background": augmentation, "values": vals})
    time.sleep(3)  #Per second calls limited

