# This is going to allow us to import python code
import ast 
import os

# Import all the needed items
from llama_index.llms.ollama import Ollama
from llama_parse import LlamaParse
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, PromptTemplate
from llama_index.core.embeddings import resolve_embed_model

# Load the env variable that the Llamacloud needs
from dotenv import load_dotenv
load_dotenv()

# Imports needed to build the agent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from prompts import context, code_parser_template

# Imports responsible for output parsing
from pydantic import BaseModel
from llama_index.core.output_parsers import PydanticOutputParser
from llama_index.core.query_pipeline import QueryPipeline # Combining multiple steps into one

# Import the code reader tool
from code_reader import code_reader

# Setup the LLM
llm = Ollama(model="llama3.1", request_timeout=3000.0)

# Setup RAG

# Goal to is to load our python file and the documentation to the model using 
# 1- We will start by loading semi-structured data or unstructured data like pdf file using llamaparse which is gonna give us better parsing results
# 2- We will then load the python file

# Creating the parser using llama parser. This parser is going to send our files to the cloud and then we will get back the parse and this is gonna give us better results when sealing with pdf
# LlamaParse can't handle python files. It is not designed to hanle python files
parser = LlamaParse(result_type= "markdown")

# Creating the file extractor 
# We are going to tell whenever you find a file with the pdf extension we are going to use the parser created to get the parse and then load it
file_extractor = { ".pdf": parser }

# Specifying the directory that the model is going to read data from (in our case its the data directory)
documents = SimpleDirectoryReader("./data", file_extractor=file_extractor).load_data()

# Now we will
# Pass these documents
# Load these files into the vector stored index
# Create vector emeddings

# local:BAAI/bge-m3 this model is responsible for creating the vector emveddings before injecting them into the vector store index
# First time the app is going to run the program is going to download this model 
embed_model = resolve_embed_model("local:BAAI/bge-m3")

# Creating our vector store index
vector_index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)

# Create the query engine that is going to be used to get some results
query_engine = vector_index.as_query_engine(llm=llm)

# Now we can use the vector_index like a question and answer bot
# Using the query engine we will be able to ask questions regarding stuff that exists in our documents
# Example
# result = query_engine.query("What are some of the toutes in the api ?")
# print(result)

# We will take the query_engine we will wrap it in a tool that we will provide to an AI agent
tools = [
    QueryEngineTool(
        query_engine=query_engine,
        metadata=ToolMetadata(
            # The name and the agent are going to tell the agent when to use this tool.
            name="api_documentation",
            description="This is going to give us documentation about an api. Use this for reading the docs for the API"
        )
    ),
    code_reader
]

# We will add an other tool that will allow the agent to load the python file


# Making the agent

# We can use multiple llms (like using a specific model for code generation)
# We can integrate multiple LLMs in our application
# code_llm = Ollama(model="llama3.1", request_timeout=3000.0)
code_llm = llm

# Setting the verbose to true this is going to show us the thoughts of the agent
agent = ReActAgent.from_tools(tools, llm=code_llm, verbose=True, context=context)



# We can convert the result from an LLM into a pydantic object
class CodeOutput(BaseModel):
    # Define the structure that we which the output to be parsed into
    # LLM result -> pydantic object
    code: str
    description: str
    filename: str

parser = PydanticOutputParser(CodeOutput)
json_prompt_str = parser.format(code_parser_template) # This parser.format() will take the string code_parser_tempalte and it will inject at the end of it the json representation of the CodeOutput. Using the tempalte afterwards in the code the model the format of the output that must be in
json_prompt_tmpl = PromptTemplate(json_prompt_str) # The resukt of the LLM that we wann parse will be injected in the {response} (check the prompts file)
output_pipeline = QueryPipeline(chain=[json_prompt_tmpl, llm]) # This LLM passed here will be responsible for parsing (in my case im just using llama3.1)

# Testing communication with the model
while (prompt := input("Enter a prompt (q to quit): ")) != "q":
    retries = 0
    while retries < 3:
        try:
            # The agent is going to take the prompt and utilize the adequat tool
            result = agent.query(prompt)
            print("result:")
            print(result)
            
            # Performing the parsing the result
            next_result = output_pipeline.run(response=result)
            print("next_result")
            print(next_result)
            
            # Cleaning the result
            cleaned_json = None
            break
        except Exception as e:
            retries += 1
            print(f"Error occured, retry #{retries}: ", e)            
            
            
    if retries >=3:
        print("Unable to process request, try again...")
        continue
    
    print("Model response")
    print(result)
    
    print("Cleaned json")
    print(cleaned_json)
    filename = cleaned_json["filename"]
    
    # If we reach this point it means everything is good to do and we can save the source code generated to a file
    try:
        with open(os.path.join("output", filename), "w") as f:
            f.write(cleaned_json["code"])
        print("Saved file", filename)
    except Exception as e:
        print("Error saving file...")
    
    
    
        
    
    
    
    