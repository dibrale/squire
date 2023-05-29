# This code is mostly copied from LangChain's modules, with a few revisions to facilitate better DuckDuckGo replies
# It is likely to change as the DuckDuckGo API code is improved

from pydantic import Field

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from duckduckgo_search import DDGS

"""Util that calls DuckDuckGo Search.

No setup required. Free.
https://pypi.org/project/duckduckgo-search/
"""
from typing import Dict, List, Optional

from pydantic import BaseModel, Extra
from pydantic.class_validators import root_validator


class DuckDuckGoCustomAPIWrapper(BaseModel):
    """Wrapper for DuckDuckGo Search API.

    Free and does not require any setup
    """
    k: int = 10
    region: Optional[str] = "wt-wt"
    safesearch: str = "Off"
    time: Optional[str] = "y"
    max_results: int = 5

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""
        try:
            from duckduckgo_search import ddg  # noqa: F401
        except ImportError:
            raise ValueError(
                "Could not import duckduckgo-search python package. "
                "Please install it with `pip install duckduckgo-search`."
            )
        return values

    def get_snippets(self, query: str) -> List[str]:
        """Run query through DuckDuckGo and return concatenated results."""

        ddgs = DDGS()
        ddgs_text_gen = ddgs.text(query, safesearch=self.safesearch)
        # ddgs_text_gen = ddgs.answers(query)

        snippets = []
        snippet_number = 0
        for r in ddgs_text_gen:
            snippets.append(str(r))
            # print('Debug print of DDG snippet: ' + str(r))
            snippet_number += 1
            if snippet_number > self.max_results:
                break

        if not snippets:
            return ["No good DuckDuckGo Search Result was found. This often happens when results are censored."]
        return snippets

    def run(self, query: str) -> str:
        snippets = self.get_snippets(query)
        return "\n".join(snippets)


class DuckDuckGoCustomRun(BaseTool):
    """Custom tool for DuckDuckGo search that does not use safe search"""

    name = "DuckDuckGo Search"
    description = (
        "A wrapper around DuckDuckGo Search. "
        "Useful for when you need to answer questions about current events. "
        "Input should be a search query."
    )
    api_wrapper: DuckDuckGoCustomAPIWrapper = Field(
        default_factory=DuckDuckGoCustomAPIWrapper
    )

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool."""
        return self.api_wrapper.run(query)

    async def _arun(
        self,
        query: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("DuckDuckGoSearch does not support async")
