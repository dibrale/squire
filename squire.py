# Squire - Use llama.cpp with LangChain tools to answer a query
# Based on the 'Custom Agent with Tool Retrieval' LangChain example
# https://python.langchain.com/en/latest/modules/agents/agents/custom_agent_with_tool_retrieval.html

# By Alexander Dibrov
# Visit me at https://github.com/dibrale/

import argparse
import os
import time

import langchain
from langchain import LLMChain, PromptTemplate
from langchain.agents import Tool, AgentExecutor, initialize_agent, AgentType
from langchain.callbacks.base import BaseCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms import LlamaCpp
from langchain.prompts import MessagesPlaceholder
from langchain.schema import AgentAction
from langchain.utilities import WikipediaAPIWrapper

from ddgtool import DuckDuckGoCustomRun


# Initialize parameters, along with any values that should not be changed in command line
params = {
    'n_ctx': 2048,
    'verbose': False
}

# Declare chat_history variable. Necessary?
chat_history = MessagesPlaceholder(variable_name="chat_history")

# Declare unified callback manager
callback_manager = BaseCallbackManager([StreamingStdOutCallbackHandler()])

no_data_string = 'No data retrieved.'

# Parse CLI arguments and load parameters
parser = argparse.ArgumentParser(
    prog='python squire.py',
    description='Use llama.cpp with LangChain tools to answer a query. '
                'Presently incorporates tools for DuckDuckGo, Arxiv and Wikipedia',
    epilog='Visit https://github.com/dibrale if you have any questions or concerns about this script.')

parser.add_argument('-q', '--question', type=str, default='question.txt',
                    help='path to a *.txt file containing your question')
parser.add_argument('-w', '--keyword_template', type=str, default='keyword_template.txt',
                    help='path to template *.txt file')
parser.add_argument('-m', '--template', type=str, default='template.txt',
                    help='path to template *.txt file')
parser.add_argument('-l', '--llama_path', type=str,
                    default="model.gguf",
                    help='path to model weights *.gguf file')
parser.add_argument('-o', '--output', type=str, default='out.txt',
                    help='path to output file for the final answer')
parser.add_argument('-p', '--top_p', type=float, default=0.7)
parser.add_argument('-r', '--repeat_penalty', type=float, default=1.1)
parser.add_argument('-k', '--top_k', type=int, default=30)
parser.add_argument('-T', '--temperature', type=float, default=0.4)
parser.add_argument('-b', '--n_batch', type=int, default=512)
parser.add_argument('-g', '--n_gpu_layers', type=int, default=0)
parser.add_argument('-t', '--n_threads', type=int, default=8)
parser.add_argument('-v', '--verbose', default=False, action='store_true', help='verbose AgentExecutor output')
params.update(vars(parser.parse_args()))

# Make extra sure that the most important parameters are not empty
assert params['template']
assert params['keyword_template']
assert params['question']
assert params['llama_path']
assert params['output']

# Ensure these parameters are in the correct format and point to actual files
if params['template'][-4:] != '.txt':
    raise ValueError('Supply a prompt template in a *.txt file with the \'-m\' option.')
elif os.path.isfile(params['template']):
    with open(params['template'], 'r') as f:
        params.update({'template_text': '\n'.join(f.readlines())})
else:
    raise FileNotFoundError(params['template'])

if params['keyword_template'][-4:] != '.txt':
    raise ValueError('Supply a keyword extraction template in a *.txt file with the \'-w\' option.')
elif os.path.isfile(params['keyword_template']):
    with open(params['keyword_template'], 'r') as f:
        params.update({'keyword_template_text': '\n'.join(f.readlines())})
else:
    raise FileNotFoundError(params['template'])

if params['question'][-4:] != '.txt':
    raise ValueError('Supply a question string or *.txt file with the \'-q\' option.')
elif os.path.isfile(params['question']):
    with open(params['question'], 'r') as f:
        params.update({'question_text': '\n'.join(f.readlines())})
else:
    raise FileNotFoundError(params['question'])

if params['llama_path'][-5:] != '.gguf':
    raise ValueError('Supply a path to a model weights *.gguf file with the \'-l\' option.')
elif not os.path.isfile(params['llama_path']):
    raise FileNotFoundError(params['llama_path'])


# 'self ask with search' agent chain initialization function
def chain_init(
        tool_wrapper, language_model, verbose=True, description="useful for when you need to ask with search"
) -> langchain.agents.initialize:
    # Function to make tools for 'self ask with search'-type agents
    def make_intermediate(wrapper) -> list[Tool]:
        return [Tool(name="Intermediate Answer", func=wrapper.run,
                     description=description)]

    return initialize_agent(
        tools=make_intermediate(tool_wrapper),
        llm=language_model,
        callback_manager=callback_manager,
        agent=AgentType.SELF_ASK_WITH_SEARCH,
        verbose=verbose
    )


# Wrapper for agents that allows them to retry in case of parse errors
def agent_wrapper(executor: AgentExecutor, question: str, iterations: int = 5, retry_wait: int = 3) -> str:
    tries_left = iterations
    exiting = False
    if tries_left < 1:
        tries_left = 0
    while tries_left > 0:
        try:
            out = executor.run(question)
            exiting = True
            return out
        except langchain.schema.OutputParserException:
            print('\nCould not parse output')
        except ConnectionResetError or ConnectionAbortedError as e:
            print('\n', e)
            if tries_left > 0:
                print('Waiting ', retry_wait, ' seconds, then retrying')
                time.sleep(retry_wait)
        except ConnectionRefusedError as e:
            print('\n', e)
            exiting = True
            tries_left = 0
        except ValueError as e:
            print('\n', e)
        finally:
            if not exiting:
                tries_left -= 1
                if tries_left > 0:
                    print('Attempts remaining: ' + str(tries_left))
    return no_data_string


# Declare LLM with parameters
llm = LlamaCpp(
    model_path=params['llama_path'],
    callback_manager=callback_manager,
    verbose=params['verbose'],
    n_ctx=params['n_ctx'],
    top_p=params['top_p'],
    top_k=params['top_k'],
    repeat_penalty=params['repeat_penalty'],
    temperature=params['temperature'],
    n_batch=params['n_batch'],
    n_threads=params['n_threads'],
    n_gpu_layers=params['n_gpu_layers'],
    echo=True,
    # stop=['Human:', 'Result:', 'Observation:']
)

# Prepare keyword prompt and initialize keyword extraction chain
keyword_prompt = PromptTemplate(input_variables=['question'], template=params['keyword_template_text'])
keyword_chain = LLMChain(
    llm=llm,
    prompt=keyword_prompt,
    verbose=params['verbose'],
    output_key='keywords')

# Prepare summary prompt and initialize summary chain
summary_prompt = PromptTemplate(input_variables=['question', 'search', 'wiki'], template=params['template_text'])
summary_chain = LLMChain(
    llm=llm,
    prompt=summary_prompt,
    verbose=params['verbose'],
    output_key='summary')

# Initialize keyword extraction chain
keywords = keyword_chain.run({'question': params['question_text']})

# Prepare keyword string
keyword_str = f"Find information about these keywords: {keywords}"

# Initialize search agent chains
search = agent_wrapper(chain_init(DuckDuckGoCustomRun(), llm, params['verbose']), keyword_str, iterations=3)
wiki = agent_wrapper(chain_init(WikipediaAPIWrapper(), llm, params['verbose'], "useful for entries from Wikipedia"),
                     keyword_str, iterations=3)

# Check for no answer condition and output accordingly, or run summary if answer is present
if search == wiki == no_data_string:
    output = 'Unable to answer the question.'
else:
    output = summary_chain.run({'question': params['question_text'], 'search': search, 'wiki': wiki})

# Write the output
with open(params['output'], 'w') as f:
    output_str = str(output).strip().strip('"').strip()
    f.write(output_str)
