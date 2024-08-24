# In this file we will define a tool that will allow us to read the context of any source code that we are going to pass to our LLM
from llama_index.core.tools import FunctionTool
import os

# We can wrap any python function as a tool that can be passed to the LLM.
# Any python function that I want the model to execute. We can do that

# This function gonna act as our tool
def code_reader_func(file_name: str):
    path = os.path.join("data", file_name)
    try:
        with open(path, "r") as f:
            content = f.read()
            return {"file_content": content}
    except Exception as e:
        return {"error": str(e)}
    
# We will wrap the function on a tool to be able to pass it to the AI agent
code_reader = FunctionTool.from_defaults(
    fn=code_reader_func,
    name="code_reader",
    description="""this tool can read the contents of code files and return 
    their results. Use this when you need to read the contents of a file""",
)