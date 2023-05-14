# Squire - Use llama.cpp with LangChain tools to answer a query
# Based on the 'Custom Agent with Tool Retrieval' LangChain example
# https://python.langchain.com/en/latest/modules/agents/agents/custom_agent_with_tool_retrieval.html

# By Alexander Dibrov
# Visit me at https://github.com/dibrale/

import argparse
import datetime
import os
import re
from typing import Callable
from typing import Union

import tiktoken
from langchain import LLMChain
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.embeddings import LlamaCppEmbeddings
from langchain.llms import LlamaCpp
from langchain.prompts import StringPromptTemplate
from langchain.schema import AgentAction, AgentFinish
from langchain.schema import Document
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import ArxivAPIWrapper
from langchain.utilities import PythonREPL
from langchain.utilities import TextRequestsWrapper
from langchain.utilities import WikipediaAPIWrapper
from langchain.vectorstores import Chroma
from requests.exceptions import MissingSchema

# Initialize parameters, along with any values that should not be changed in command line
params = {
    'n_ctx': 2048,
    'do_local_date_time': True
}

# Parse CLI arguments and load parameters
parser = argparse.ArgumentParser(
                    prog='python squire.py',
                    description='Use llama.cpp with LangChain tools to answer a query. '
                                'Presently incorporates tools for '
                                'DuckDuckGo, Arxiv, Wikipedia, Requests and PythonREPL.',
                    epilog='Visit https://github.com/dibrale if you have any questions or concerns about this script.')

parser.add_argument('-q', '--question', type=str, default='question.txt',
                    help='path to a *.txt file containing your question')
parser.add_argument('-m', '--template', type=str, default='template.txt',
                    help='path to template *.txt file')
parser.add_argument('-l', '--llama_path', type=str,
                    default="ggml-model-q4_0.bin",
                    help='path to ggml model weights *.bin file')
parser.add_argument('-o', '--output', type=str, default='out.txt',
                    help='path to output file for the final answer')
parser.add_argument('-p', '--top_p', type=float, default=0.5)
parser.add_argument('-k', '--top_k', type=float, default=40)
parser.add_argument('-T', '--temperature', type=float, default=0.2)
parser.add_argument('-b', '--n_batch', type=float, default=512)
parser.add_argument('-t', '--n_threads', type=float, default=6)
parser.add_argument('-v', '--verbose', default=False, action='store_true', help='verbose AgentExecutor output')
params.update(vars(parser.parse_args()))

# Make extra sure that the most important parameters are not empty
assert params['template']
assert params['question']
assert params['llama_path']
assert params['output']

# Ensure these parameters are in the correct format and point to actual files
if params['template'][-4:] != '.txt':
    raise ValueError('Supply a prompt template in a *.txt file with the \'--template\' option.')
elif os.path.isfile(params['template']):
    with open(params['template'], 'r') as f:
        params.update({'template_text': '\n'.join(f.readlines())})
else:
    raise FileNotFoundError(params['template'])

if params['question'][-4:] != '.txt':
    raise ValueError('Supply a question string or *.txt file with the \'-q\' option.')
elif os.path.isfile(params['question']):
    with open(params['question'], 'r') as f:
        params.update({'question_text': '\n'.join(f.readlines())})
else:
    raise FileNotFoundError(params['question'])

if params['llama_path'][-4:] != '.bin':
    raise ValueError('Supply a path to a ggml model weights *.bin file with the \'-l\' option.')
elif not os.path.isfile(params['llama_path']):
    raise FileNotFoundError(params['llama_path'])


# Some timestamp functions to help the LLM orient for current event lookups
def long_date(now=datetime.datetime.now()):
    return f"""{now.strftime('%A')}, {now.strftime('%B')} {now.strftime('%d')}, {now.strftime('%Y')}"""


def time(now=datetime.datetime.now()):
    return f"""{now.strftime('%H')}:{now.strftime('%M')}"""


# Mock tool that lets us scold the LLM in case of a malformed reply, giving it a chance to improve its next output
def non_parse_fcn(inp: str) -> str:
    return 'Your reply could not be parsed. Try again. If you have an answer, start your reply with \'Final Answer: \''


# Mock tool to scold the LLM if it does not provide an Action
def unclear_tool_fcn(inp: str) -> str:
    return 'No Action was provided. Try again. If you wish to use a tool, write a \'Action: \' followed by the tool name.'


# Mock tool that lets us scold the LLM in case of a malformed reply, giving it a chance to improve its next output
def unclear_input_fcn(inp: str) -> str:
    return 'No Action Input was provided. Try again. If you wish to use a tool, write a \'Action Input: \' followed by your desired input for it.'


# Better wrapper for Requests
def squire_requests_wrapper(url):
    try:
        TextRequestsWrapper().get(url)
    except MissingSchema:
        return 'You need to enter a URL as your Action Input if you are using Requests. Try again.'


non_parse_tool = Tool(
    name="NotParsed",
    func=non_parse_fcn,
    description="This is a dummy option. Do not use."
)

unclear_tool = Tool(
    name="UnclearTool",
    func=unclear_tool_fcn,
    description="This is a dummy option. Do not use."
)

unclear_input_tool = Tool(
    name="UnclearInput",
    func=unclear_input_fcn,
    description="This is a dummy option. Do not use."
)

# Define the rest of the tools
search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="Search",
    func=search.run,
    description="Responds to general queries of all kinds. Good for current events and conditions."
)

arxiv = ArxivAPIWrapper()
arxiv_tool = Tool(
    name="Arxiv",
    func=arxiv.run,
    description="Returns information about scientific articles. Not good for current events and conditions."
)

wikipedia = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia.run,
    description="Returns introductory encyclopedic information about various topics. Less useful for current events. "
                "Not good for current conditions"
)

requests = TextRequestsWrapper()
requests_tool = Tool(
    name="Requests",
    func=squire_requests_wrapper,
    description="Takes in a URL and fetches raw data from it. Especially useful for plaintext-heavy web content."
)

repl = PythonREPL()
repl_tool = Tool(
    name="PythonREPL",
    func=repl.run,
    description="Action Input must be valid Python code without syntax errors. Instruct the tool to print out an answer if you use it."
)

# Tool lists
ALL_REAL_TOOLS = [search_tool] \
            + [arxiv_tool] \
            + [wikipedia_tool] \
            + [requests_tool] \
            + [repl_tool] \

ALL_TOOLS = ALL_REAL_TOOLS \
            + [non_parse_tool] \
            + [unclear_tool] \
            + [unclear_input_tool]

# Document, embedding and db definitions.
docs = [Document(page_content=t.description, metadata={"index": i}) for i, t in enumerate(ALL_TOOLS)]
llama_embed = LlamaCppEmbeddings(model_path=params['llama_path'])
vector_store = Chroma.from_documents(documents=docs, embedding=llama_embed)
retriever = vector_store.as_retriever()


# Tool retrieval function
def get_tools(query):
    tools_docs = retriever.get_relevant_documents(query)
    return [ALL_TOOLS[d.metadata["index"]] for d in tools_docs]


# Extract tool name from 'Action: ' substring, even if the model decides to be a bit too verbose.
def parse_tool(txt: str):
    tool_frequency = {}

    for tool in tool_names:
        tool_frequency.update({tool: len(re.findall(tool, txt))})
    most_frequent_tool = max(tool_frequency, key=tool_frequency.get)

    if not most_frequent_tool or len(most_frequent_tool) == 0:
        return 'Unclear'
    else:
        return most_frequent_tool


# Truncate context for Llama using tiktoken to estimate length
def llama_squeeze(txt, max_tokens=2048, encoding="gpt-3.5-turbo"):
    enc = tiktoken.encoding_for_model(encoding)
    encoded_txt = enc.encode(txt, disallowed_special=set())
    tokens_start = max(0, len(encoded_txt) - max_tokens)
    truncated_encoded_txt = encoded_txt[tokens_start:]
    decoded_txt = enc.decode(truncated_encoded_txt)
    return decoded_txt


# Set up a prompt template
class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools_getter: Callable

    def format(self, **kwargs) -> str:
        # Get formatted intermediate steps and append the timestamp
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            if params['do_local_date_time']:
                thoughts += f"\nCurrent Date: {long_date()}"
                thoughts += f"\nCurrent Local Time: {time()}"
                params['do_local_date_time'] = False
            thoughts = llama_squeeze(thoughts, 500)
            thoughts += f"\nObservation: {llama_squeeze(observation, 800)}\nThought: "

        # Set the agent_scratchpad variable to that value
        kwargs["agent_scratchpad"] = thoughts

        # Create a tools variable and tool names list from the list of tools provided
        tools = self.tools_getter(kwargs["input"])
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in tools])

        return self.template.format(**kwargs)


prompt = CustomPromptTemplate(
    template=params['template_text'],
    tools_getter=get_tools,
    input_variables=["input", "intermediate_steps"]
)


# Our version of the output parser
class CustomOutputParser(AgentOutputParser):

    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:

        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )

        # Parse out the action and action input
        regex = r"\n*\s*\d*Action\s*\d*\s*:\s*\d*\s*(?P<ACTION>(\w*))\n*\s*\d*(?:Action)*\n*\s*\d*Input\s*\d*\s*:[\s]*(?P<INPUT>(.*))"
        match = re.search(regex, llm_output, re.DOTALL)

        # Instead of raising an error due to malformed output, tell our LLM how to do better next time
        if not match:
            action = "NonParseTool"
            action_input = ''

        # If the format is good, check the Action and Action Input.
        else:

            action_text = match.group('ACTION').strip()
            action_input = match.group('INPUT')

            # First, check the action text for an interpretable action
            action = parse_tool(action_text)

            # Next, check that the action input is not empty
            if action_input == '':
                action = 'UnclearInput'

            # Fix small input mistakes for specific tools. For now, just matters for PythonREPL
            if action == "PythonREPL":
                regex = r".*print((.*)).*"
                match = re.search(regex, action_input, re.DOTALL)
                if not match:
                    action_input = f'print({action_input})'

        # Return the action and action input
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)


# Instantiate the output parser, callback manager, LLM, LLM chain, tool names, agent and agent executor
output_parser = CustomOutputParser()
callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path=params['llama_path'],
    callback_manager=callback_manager,
    verbose=params['verbose'],
    n_ctx=params['n_ctx'],
    top_p=params['top_p'],
    top_k=params['top_k'],
    temperature=params['temperature'],
    n_batch=params['n_batch'],
    n_threads=params['n_threads']
)
llm.client.verbose = params['verbose']
llm_chain = LLMChain(llm=llm, prompt=prompt, verbose=params['verbose'])

# Fake tools names not added to tool_names
tool_padding = ['']*(len(ALL_TOOLS)-len(ALL_REAL_TOOLS))
real_tool_names = [tool.name for tool in ALL_REAL_TOOLS]
tool_names = real_tool_names + tool_padding

agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\nObservation:"],
    allowed_tools=tool_names,
    verbose=params['verbose']
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=ALL_REAL_TOOLS, verbose=params['verbose'], max_iterations=3, early_stopping_method='force')

# Run the agent executor
output = agent_executor.run(params['question_text'])

# Write the output
with open(params['output'], 'w') as f:
    output.strip().strip('"').strip()
    f.write(output)
