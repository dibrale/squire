# Squire - Use llama.cpp with LangChain tools to answer a query
# Based on the 'Custom Agent with Tool Retrieval' LangChain example
# https://python.langchain.com/en/latest/modules/agents/agents/custom_agent_with_tool_retrieval.html

# By Alexander Dibrov
# Visit me at https://github.com/dibrale/

import argparse
import os

import langchain
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import SequentialChain
from langchain.llms import LlamaCpp
from langchain.prompts import MessagesPlaceholder
from langchain.schema import AgentAction
from langchain.tools import DuckDuckGoSearchRun
from langchain.utilities import ArxivAPIWrapper, WikipediaAPIWrapper

# Initialize parameters, along with any values that should not be changed in command line
params = {
    'n_ctx': 2048,
    'verbose': False
}

no_data_string = 'No data retrieved.'

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
                    # default="ggml-model-q4_0.bin",
                    default='ggml-model-q4_0.bin',
                    help='path to ggml model weights *.bin file')
parser.add_argument('-o', '--output', type=str, default='out.txt',
                    help='path to output file for the final answer')
parser.add_argument('-p', '--top_p', type=float, default=0.8)
parser.add_argument('-r', '--repeat_penalty', type=float, default=1.1)
parser.add_argument('-k', '--top_k', type=float, default=30)
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


# 'self ask with search' agent chain initialization function
def chain_init(tool_wrapper, language_model, verbose=True) -> langchain.agents.initialize:

    # Function to make tools for 'self ask with search'-type agents
    def make_intermediate(wrapper) -> list[Tool]:
        return [Tool(name="Intermediate Answer", func=wrapper.run,
                     description="useful for when you need to ask with search")]

    return initialize_agent(
        tools=make_intermediate(tool_wrapper),
        llm=language_model,
        agent=AgentType.SELF_ASK_WITH_SEARCH,
        verbose=verbose)


# Wrapper for agents that allows them to retry in case of parse errors
def agent_wrapper(executor: AgentExecutor, question: str, iterations: int = 3) -> str:
    tries_left = iterations
    if tries_left < 1:
        tries_left = 0
    while tries_left > 0:
        try:
            out = executor.run(question)
            return out
        except langchain.schema.OutputParserException:
            print('\nCould not parse output')
            tries_left -= 1
            if tries_left > 0:
                print('Attempts remaining: ' + str(tries_left))
    return no_data_string


# Declare chat_history variable. Necessary?
chat_history = MessagesPlaceholder(variable_name="chat_history")

# Declare LLM with parameters
llm = LlamaCpp(
    model_path=params['llama_path'],
    callback_manager=BaseCallbackManager([StreamingStdOutCallbackHandler()]),
    verbose=params['verbose'],
    n_ctx=params['n_ctx'],
    top_p=params['top_p'],
    top_k=params['top_k'],
    repeat_penalty=params['repeat_penalty'],
    temperature=params['temperature'],
    n_batch=params['n_batch'],
    n_threads=params['n_threads'],
    echo=True,
    # stop=['Human:', 'Result:', 'Observation:']
)

# Prepare summary prompt and initialize summary chain
summary_prompt = PromptTemplate(input_variables=['search', 'wiki', 'arxiv'], template=params['template_text'])
summary_chain = LLMChain(
    llm=llm,
    prompt=summary_prompt,
    verbose=params['verbose'],
    output_key='summary')

# Initialize search agent chains
search = agent_wrapper(chain_init(DuckDuckGoSearchRun(), llm, params['verbose']), params['question_text'])
wiki = agent_wrapper(chain_init(WikipediaAPIWrapper(), llm, params['verbose']), params['question_text'])
arxiv = agent_wrapper(chain_init(ArxivAPIWrapper(), llm, params['verbose']), params['question_text'])

# Initialize overall chain
overall_chain = SequentialChain(
    chains=[summary_chain],
    input_variables=["search", "wiki", "arxiv"],
    output_variables=['summary'],
    verbose=params['verbose'])

# Check for no answer condition and output accordingly, or run summary if answer is present
if search == wiki == arxiv == no_data_string:
    output = 'Unable to answer the question.'
else:
    output = overall_chain.run({'search': search, 'wiki': wiki, 'arxiv': arxiv})

# Write the output
with open(params['output'], 'w') as f:
    output_str = str(output).strip().strip('"').strip()
    f.write(output_str)
