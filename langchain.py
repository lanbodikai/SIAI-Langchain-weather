import json
from typing import List, Optional
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms.base import LLM
from langchain_community.llms import Ollama
import ollama  # Ensure the ollama Python package is installed
from langchain.utilities import DuckDuckGoSearchAPIWrapper

# -------------------------------------------
# Step 1: Extract parameters using Llama 3.2 via Ollama
# -------------------------------------------
llm_llama = Ollama(model="llama3.2")  # Use "llama3.2" if that’s your model name

prompt_template_llama = """
Extract the following parameters from the given query:
- **Location**: Where is the event happening?
- **Date**: What is the date mentioned?
- **Time**: What time is specified?

If a parameter is not mentioned, set its value to "null".

Examples:
1. Query: "Find events in New York on January 5, 2023 at 3 PM."
   JSON:
   {{
     "Location": "New York",
     "Date": "January 5, 2023",
     "Time": "3 PM"
   }}

2. Query: "What's the weather in England yesterday?"
   JSON:
   {{
     "Location": "England",
     "Date": "yesterday",
     "Time": "null"
   }}

Query: "{query}"

Provide the answer in JSON format:
{{
  "Location": "<extracted location or null>",
  "Date": "<extracted date or null>",
  "Time": "<extracted time or null>"
}}
"""

chain_llama = LLMChain(
    llm=llm_llama,
    prompt=PromptTemplate.from_template(prompt_template_llama)
)

user_query = input("Please enter your query: ")
response_llama = chain_llama.invoke({"query": user_query})
print("\n[LLAMA 3.2 RAW OUTPUT]:")
print(response_llama["text"])

try:
    parameters = json.loads(response_llama["text"].strip())
except json.JSONDecodeError:
    parameters = {"Location": "null", "Date": "null", "Time": "null"}

# -------------------------------------------
# Step 2: Check for missing parameters and ask user to fill them in
# -------------------------------------------
for key in parameters:
    if parameters[key].strip().lower() == "null":
        new_val = input(f"The parameter '{key}' was not provided. Please enter a value for {key}: ")
        parameters[key] = new_val.strip()

print("\n[FINAL PARAMETERS]:")
print(json.dumps(parameters, indent=2))

# -------------------------------------------
# Step 3: Use a web search tool to retrieve data based on the parameters
# -------------------------------------------
search_query = f"weather in {parameters['Location']} on {parameters['Date']} at {parameters['Time']}"
print("\n[SEARCH QUERY]:", search_query)

search_tool = DuckDuckGoSearchAPIWrapper()
search_results = search_tool.run(search_query)
print("\n[RAW SEARCH RESULTS]:")
print(search_results)

# -------------------------------------------
# Step 4: Convert search results into human-friendly text using DeepSeek via Ollama
# -------------------------------------------
class OllamaDeepSeekLLM(LLM):
    model: str

    class Config:
        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        return "ollama_deepseek"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = ollama.chat(self.model, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

llm_deepseek = OllamaDeepSeekLLM(model="deepseek-r1:8b")

prompt_template_deepseek = """
Convert the following raw search results into a clear, human-friendly summary:

Raw Data:
{search_results}

Summary:
"""
chain_deepseek = LLMChain(
    llm=llm_deepseek,
    prompt=PromptTemplate.from_template(prompt_template_deepseek)
)

response_deepseek = chain_deepseek.invoke({"search_results": search_results})
final_summary = response_deepseek["text"].strip()

print("\n[FINAL HUMAN-FRIENDLY SUMMARY]:")
print(final_summary)