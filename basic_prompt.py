from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import numpy as np
from benchmark_functions import forrester
import time


template = """You are an optimization researcher tasked to minimize the value of a function
 with 1 input variable x. 
Current sampled points for 1 variable (x) with their respective function values in csv format are:
x, function_value
{values}

Give me a new (x) value that satisfies the following: 
(a) new x is different from all above, 
(b) new x value lies in the range 0 to 1,
(c) has a function value lower than the above function loss values and 
(d) result in a rapid convergence towards the value of x that results in the global minimum of the function. 
Do not write code or any explanation. The output must end with numerical value for (x) only."""

prompt_template = ChatPromptTemplate.from_template(template)
vals = f" 0.05, {forrester(0.05)} \n 0.95, {forrester(0.95)} \n 0.5, {forrester(0.5)} \n"
prompt = prompt_template.invoke({"values": vals})

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

num_steps = 10
for i in range(num_steps):
    result = model.invoke(prompt)
    x = float(result.content)
    print(f"New Sample: {x}, {forrester(x)}")
    vals += f" {x}, {forrester(x)} \n"
    prompt = prompt_template.invoke({"values": vals})
    time.sleep(3)  #Per second calls limited

print(vals)
