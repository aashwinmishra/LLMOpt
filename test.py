from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import numpy as np
from benchmark_functions import ackley
import time


template = """You are an optimization researcher tasked to minimize the value of a function
 with 2 input variables: x and y. This function's values are almost flat with multiple local minima in most of the (x, y) space. 
 There is one global minimum that has a function value of 0. Current sampled points for the input variables (x,y) with their respective function values in csv format are:
x, y, function_value
{values}

Give me a new (x, y) value that satisfies the following:
(a) new (x, y) is different from all above,
(b) new (x, y) values lie in the range -1 to 1 for both x and y,
(c) has a function value lower than the above function loss values and
(d) result in a rapid convergence towards the value of x that results in the global minimum of the function.
Do not write code or any explanation. The output must end with numerical value for (x, y) only."""

prompt_template = ChatPromptTemplate.from_template(template)

vals = f" 0.9, 0.85, {ackley([0.9, 0.85])} \n -0.7, 0.9, {ackley([-0.7, 0.9])} \n 0.9, -0.7, {ackley([0.9, -0.7])} \n"

prompt = prompt_template.invoke({"values": vals})

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
chain = prompt_template | model | StrOutputParser()
result = chain.invoke({"values": vals})
temp = np.array(list(map(float, result.split(","))))
print(ackley(temp))


# num_steps = 10
#
# for i in range(num_steps):
#     result = model.invoke(prompt)
#     x = float(result.content)
#     print(f"New Sample: {x}, {forrester(x)}")
#     vals += f" {x}, {forrester(x)} \n"
#     prompt = prompt_template.invoke({"values": vals})
#     time.sleep(3)  #Per second calls limited
#
# print(vals)
#
#
