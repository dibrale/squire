![squire_smol](https://user-images.githubusercontent.com/108030031/235379536-84e1b66b-903f-4026-a942-31b0ab885ff8.png)

# Squire - Get Answers with Llama

Use [llama.cpp](https://github.com/ggerganov/llama.cpp) with [LangChain](https://docs.langchain.com/docs/) tools to answer a query. This script is based on the [Custom Agent with Tool Retreival](https://python.langchain.com/en/latest/modules/agents/agents/custom_agent_with_tool_retrieval.html) example from LangChain, with prompt engineering and output parsing modifications to better accomodate Llama models. Runs as a command line script.

## Installation

1. Clone this repository and navigate into its directory
2. Install dependencies, i.e. using `pip install -r requirements.txt`
3. Download a CPU inference checkpoint compatible with llama.cpp
4. Edit question.txt with your query
5. Run the script using `python squire.py`, followed by command line options.

## Usage

No command line options are required id the default model (wizardLM-7B.GGML.q4_2.bin), question file and template file are present in the script directory. 
Command line options for external files:

| Option  | Description | Default |
| ------------- | ------------- | ------------- |
| -q --question | path to a *.txt file containing your question | question.txt |
| -m --template | path to template *.txt file | template.txt |
| -l --llama_path| path to ggml model weights *.bin file | wizardLM-7B.GGML.q4_2.bin |
| -o --output | path to output file for the final answer | out.txt |

There are also options to control what parameters the LangChain `LlamaCpp()` forwards to llama.cpp:

| Option  | Default |
| ------------- | ------------- |
| -p --top_p | 0.95 |
| -k --top_k | 40 |
| -T --temperature | 0.2 |
| -b --n_batch | 512 |
| -t -- n_threads | 6 |

Miscellaneous:

| Option  | Description |
| ------------- | ------------- |
| -v --verbose | verbose output (partially working) |

## Operation

Squire will ingest default or provided parameters. Assuming this is successful, it will use LangChain to run the provided model using llama.cpp for encoding and inference. The model will choose a tool with which to search for an answer. The available tools have been selected because they do not require an API key and are free to use. Currently available tools:

- DuckDuckGo Search
- Arxiv
- Wikipedia
- Requests
- PythonREPL

It may take the model several cycles of queries and attempted queries before it obtains satisfactory data to use in a final answer. Once it does obtain adequate data, Squire will write it to the output file and in command line. Squire has been tested with [wizardLM-7B](https://huggingface.co/TheBloke/wizardLM-7B-GGML) and [llama-30b-supercot-ggml](https://huggingface.co/gozfarb/llama-30b-supercot-ggml), giving answers to the best of its ability. Note that the quality of the output and efficiency of operation will be greatly affected by your choice of local language model.

## Afterword
  
I hope you find this tool both useful and simple to use. Please let me know if you encounter any issues or have any suggestions.
