"""
title: Web Search and Crawl
description: Search and Crawls the web using SearXNG, OpenWebUI Native Search, and Crawl4AI. Extracts content from URLs using a self-hosted Crawl4AI instance, optionally researching using Crawl4AI Deep Research.
author: lexiismadd
author_url: https://github.com/lexiismadd
funding_url: https://github.com/open-webui
version: 2.8.3
license: MIT
requirements: aiohttp, loguru, crawl4ai, orjson, tiktoken
"""

# region ── Imports ────────────────────────────────────────────────────────────

import os
import re
import traceback
import requests
import orjson
import tiktoken
import aiohttp
import asyncio
from urllib.parse import parse_qs, urlparse, quote
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Union, Callable, Literal
from loguru import logger
from crawl4ai import (
    BestFirstCrawlingStrategy,
    CrawlerRunConfig,
    DefaultTableExtraction,
    KeywordRelevanceScorer,
    LLMConfig,
    BrowserConfig,
    CacheMode,
    DefaultMarkdownGenerator,
    LLMExtractionStrategy,
)
from crawl4ai.content_filter_strategy import PruningContentFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# OpenWebUI imports for native search
try:
    from open_webui.main import Request, app  # type: ignore
    from open_webui.models.users import UserModel, Users  # type: ignore
    from open_webui.routers.retrieval import SearchForm, process_web_search  # type: ignore

    NATIVE_SEARCH_AVAILABLE = True
except ImportError:
    NATIVE_SEARCH_AVAILABLE = False
    logger.warning(
        "OpenWebUI native search not available - install requirements or check OpenWebUI version"
    )

# endregion

# region ── Models ─────────────────────────────────────────────────────────────


class ArticleData(BaseModel):
    topic: str
    summary: str


class ResearchCrawlMode:
    """Enumeration of research crawling modes."""

    PSEUDO_ADAPTIVE = "pseudo_adaptive"
    LLM_GUIDED = "llm_guided"
    BFS_DEEP = "bfs_deep"
    RESEARCH_FILTER = "research_filter"


# endregion


class Tools:

    # region ── Valves ─────────────────────────────────────────────────────────

    class Valves(BaseModel):
        INITIAL_RESPONSE: str = Field(
            title="Initial delta response",
            default="I just need to do a search online to get some more info, I'll get back to you in a minute or so with a response if thats ok with you...",
            description="The response the tool will post in the chat window when it starts its search and crawl. Set as blank for no response.",
        )
        USE_NATIVE_SEARCH: bool = Field(
            title="Use Native Search",
            default=True,
            description="Use OpenWebUI's native web search (in addition to or instead of SearXNG).",
        )
        SEARCH_WITH_SEARXNG: bool = Field(
            title="Search with SearXNG",
            default=False,
            description="Use SearXNG for gathering additional URLs for crawling.",
        )
        SEARXNG_BASE_URL: str = Field(
            title="SearXNG Search URL",
            default="http://searxng:8888/search?format=json&q=<query>",
            description="The full URL for your SearXNG API instance. Insert <query> where the search terms should go.",
        )
        SEARXNG_API_TOKEN: str = Field(
            title="SearXNG API Token",
            default="",
            description="The API token or Secret for your SearXNG instance.",
        )
        SEARXNG_METHOD: Literal["GET", "POST"] = Field(
            title="SearXNG HTTP Method",
            default="GET",
            description="HTTP method to use for SearXNG API calls (GET or POST).",
        )
        SEARXNG_TIMEOUT: int = Field(
            title="SearXNG Timeout",
            default=30,
            description="The timeout (in seconds) for SearXNG API requests.",
        )
        SEARXNG_MAX_RESULTS: int = Field(
            title="SearXNG Max Results",
            default=10,
            description="The maximum number of results to return from SearXNG.",
        )
        CRAWL4AI_BASE_URL: str = Field(
            title="Crawl4AI Base URL",
            default="http://crawl4ai:11235",
            description="The base URL for your Crawl4AI instance.",
        )
        CRAWL4AI_USER_AGENT: str = Field(
            title="Crawl4AI User Agent",
            default="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.1.2.3 Safari/537.36",
            description="Custom User-Agent string for Crawl4AI.",
        )
        CRAWL4AI_TIMEOUT: int = Field(
            title="Crawl4AI Timeout",
            default=60,
            description="The timeout (in seconds) for Crawl4AI requests.",
        )
        CRAWL4AI_BATCH: int = Field(
            title="Crawl4AI Batch",
            default=5,
            description="The number of URLs to send to Crawl4AI per batch. If more than this number of URLs are found in total, the tool will send them to Crawl4AI in batches of this number. Useful for reducing the tokens used by the LLM per crawl.",
        )
        CRAWL4AI_MAX_URLS: int = Field(
            title="Crawl4AI Maximum URLs to crawl",
            default=20,
            description="The maximum number of URLs to crawl with Crawl4AI.",
        )
        CRAWL4AI_EXTERNAL_DOMAINS: bool = Field(
            title="Crawl External Domains",
            default=False,
            description="Allow Crawl4AI to crawl external/additional URL domains.",
        )
        CRAWL4AI_EXCLUDE_DOMAINS: str = Field(
            title="Excluded Domains",
            default="",
            description="Comma-separated list of external domains to exclude from crawling.",
        )
        CRAWL4AI_EXCLUDE_SOCIAL_MEDIA_DOMAINS: str = Field(
            title="Excluded Social Media Domains",
            default="facebook.com,twitter.com,x.com,linkedin.com,instagram.com,pinterest.com,tiktok.com,snapchat.com,reddit.com",
            description="Comma-separated list of social media domains to exclude from crawling.",
        )
        CRAWL4AI_EXCLUDE_IMAGES: Literal["None", "External", "All"] = Field(
            title="Exclude Images",
            default="None",
            description="Exclude images from crawling (None, External, All).",
        )
        CRAWL4AI_WORD_COUNT_THRESHOLD: int = Field(
            title="Word Count Threshold",
            default=200,
            description="The minimum word count threshold for content to be included.",
        )
        CRAWL4AI_TEXT_ONLY: bool = Field(
            title="Text Only",
            default=False,
            description="Only extract text content, excluding images and other media. (Disables crawling and displaying media in the chat)",
        )
        CRAWL4AI_DISPLAY_MEDIA: bool = Field(
            title="Display Media in Chat",
            default=True,
            description="Display images and videos as clickable links in the chat window.",
        )
        CRAWL4AI_MAX_MEDIA_ITEMS: int = Field(
            title="Max Media Items to Display",
            default=5,
            description="Maximum number of images/videos to display (0 = unlimited).",
        )
        CRAWL4AI_DISPLAY_THUMBNAILS: bool = Field(
            title="Display images as thumbnails",
            default=False,
            description="Display images as thumbnails in the chat window. Turn off to display images full-sized.",
        )
        CRAWL4AI_THUMBNAIL_SIZE: int = Field(
            title="Image thumbnail size",
            default=200,
            description="Image thumbnail size (in px) square.  eg, setting 200 will mean thumbnails are 200x200px in size. Ignored if 'Display images as thumbnails' is off.",
        )
        CRAWL4AI_MIN_IMAGE_SCORE: int = Field(
            title="Min Image Score To Include",
            default=6,
            ge=0,
            le=10,
            description="Minimum image score from Crawl4AI to consider including in the response. Min 0, Max 10.",
        )
        CRAWL4AI_VALIDATE_IMAGES: bool = Field(
            title="Validate Image Links",
            default=True,
            description="Validate any image links to make sure they are accessible.",
        )
        CRAWL4AI_MAX_TOKENS: int = Field(
            title="Max Tokens used by Crawl4AI",
            default=0,
            description="Maximum tokens to use for the web search content response. Set to 0 for unlimited.",
        )
        LLM_BASE_URL: str = Field(
            title="LLM Base URL",
            default="https://openrouter.ai/api/v1",
            description="The base URL for your preferred OpenAI-compatible LLM.",
        )
        LLM_API_TOKEN: str = Field(
            title="LLM API Token",
            default="",
            description="Optional API Token for your preferred OpenAI-compatible LLM.",
        )
        LLM_PROVIDER: str = Field(
            title="LLM Provider and model",
            default="openrouter/@preset/default",
            description="The LLM provider and model to use (see https://docs.crawl4ai.com/core/browser-crawler-config/#3-llmconfig-essentials).",
            examples=[
                "openai/gpt-4o",
                "ollama/llama-3-70b",
                "openrouter/@preset/default",
                "azure/gpt-4o",
                "anthropic/claude-2",
            ],
        )
        LLM_TEMPERATURE: float = Field(
            title="LLM Temperature",
            default=0.3,
            description="The temperature to use for the LLM.",
        )
        LLM_INSTRUCTION: str = Field(
            title="LLM Extraction Instruction",
            default="""Focus on extracting the core content. Summarize lengthy sections into concise points
            Include:
            - Key concepts and explanations
            - Important examples
            - Critical details that enhance understanding
            - Data from tables that support the main content
            - Any relevant data snippets
            Exclude:
            - Navigation elements
            - Sidebars
            - Footer content
            - Marketing or promotional material
            - Advertisements
            - User comments
            - Any other non-essential information
            Format the output as clean markdown with proper code blocks and headers.
            """,
            description="The instruction to use for the LLM when extracting from the webpage.",
        )
        LLM_MAX_TOKENS: int = Field(
            title="LLM Max Tokens",
            default=4096,
            description="The maximum number of tokens to use for the LLM.",
        )
        LLM_TOP_P: float = Field(
            title="LLM Top P",
            default=None,
            description="The top_p value to use for the LLM.",
        )
        LLM_FREQUENCY_PENALTY: float = Field(
            title="LLM Frequency Penalty",
            default=None,
            description="The frequency penalty to use for the LLM.",
        )
        LLM_PRESENCE_PENALTY: float = Field(
            title="LLM Presence Penalty",
            default=None,
            description="The presence penalty to use for the LLM.",
        )
        MORE_STATUS: bool = Field(
            title="More status updates",
            default=False,
            description="Show more status updates during web search and crawl",
        )
        DEBUG: bool = Field(
            title="Debug logging",
            default=False,
            description="Enable detailed debug logging",
        )

    # endregion

    # region ── User Valves ────────────────────────────────────────────────────

    class UserValves(BaseModel):
        """Per-user configurable options for Research Mode and crawling strategies."""

        SEARXNG_MAX_RESULTS: int = Field(
            title="SearXNG Max Results",
            default=None,
            description="Per-user maximum results from SearXNG.",
        )
        CRAWL4AI_MAX_URLS: int = Field(
            title="Crawl4AI Maximum URLs to crawl",
            default=None,
            description="Per-user maximum URLs to crawl.",
        )
        CRAWL4AI_DISPLAY_MEDIA: bool = Field(
            title="Display Media in Chat",
            default=None,
            description="Per-user media display setting.",
        )
        CRAWL4AI_MAX_MEDIA_ITEMS: int = Field(
            title="Max Media Items to Display",
            default=None,
            description="Per-user max media items.",
        )
        CRAWL4AI_DISPLAY_THUMBNAILS: bool = Field(
            title="Display images as thumbnails",
            default=None,
            description="Per-user thumbnail setting.",
        )
        CRAWL4AI_THUMBNAIL_SIZE: int = Field(
            title="Image thumbnail size",
            default=None,
            description="Per-user thumbnail size.",
        )
        RESEARCH_MODE: bool = Field(
            default=False,
            description="Enable research mode (deep crawling).",
        )
        RESEARCH_CRAWL_MODE: Literal[
            "pseudo_adaptive", "llm_guided", "bfs_deep", "research_filter"
        ] = Field(
            default="pseudo_adaptive",
            description="Crawling strategy for research mode.",
        )
        RESEARCH_KEYWORD_WEIGHT: float = Field(
            default=0.7,
            description="Keyword relevance weight for research mode.",
        )
        RESEARCH_MAX_DEPTH: int = Field(
            default=2,
            le=10,
            description="Maximum crawl depth for research mode.",
        )
        RESEARCH_MAX_PAGES: int = Field(
            default=15,
            le=25,
            description="Maximum pages to crawl in research mode.",
        )
        RESEARCH_BATCH_SIZE: int = Field(
            default=5,
            description="Batch size for research crawling.",
        )
        RESEARCH_LLM_LINK_SELECTION: bool = Field(
            default=True,
            description="Use LLM to select next links in llm_guided mode.",
        )
        RESEARCH_INCLUDE_EXTERNAL: bool = Field(
            default=False,
            description="Allow external domains in research mode.",
        )

    # endregion

    # region ── Init & Configuration ───────────────────────────────────────────

    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

        self._configure()

        if self.valves.SEARCH_WITH_SEARXNG and self.valves.SEARXNG_BASE_URL:
            searxng_parsed_url = urlparse(self.valves.SEARXNG_BASE_URL)
            searxng_parsed_url_query = parse_qs(searxng_parsed_url.query)
            if "q" not in searxng_parsed_url_query:
                searxng_parsed_url_query["q"] = ["<query>"]
            if "format" in searxng_parsed_url_query:
                if searxng_parsed_url_query["format"][0] != "json":
                    searxng_parsed_url_query["format"][0] = "json"
            reconstructed_query = "&".join(
                [f"{key}={value[0]}" for key, value in searxng_parsed_url_query.items()]
            )
            self.valves.SEARXNG_BASE_URL = (
                f"{searxng_parsed_url.scheme}://{searxng_parsed_url.netloc}"
                f"{searxng_parsed_url.path}?{reconstructed_query}"
            )

        # Define tools for better LLM integration
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_and_crawl",
                    "description": "Search the web and crawl the resulting pages to extract detailed content with images and videos. Use this for current events, news, research, or any information that needs web search and detailed content extraction. The user can optionally provide specific URLs to include in the crawl. When research_mode is enabled, multiple crawling strategies are available including pseudo-adaptive keyword scoring, LLM-guided link selection, BFS deep crawling, and research filtering.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query (e.g., 'latest AI developments', 'Python tutorial')",
                            },
                            "urls": {
                                "type": "array",
                                "description": "Optional list of specific URLs to crawl in addition to search results",
                                "items": {"type": "string"},
                                "default": [],
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of search results to crawl (default uses valve setting)",
                                "default": None,
                            },
                            "research_mode": {
                                "type": "boolean",
                                "description": "Enables Research Mode which performs deeper web crawling using advanced strategies. When enabled, the LLM can also specify a research_crawl_mode parameter to choose the crawling strategy.",
                                "default": False,
                            },
                            "research_crawl_mode": {
                                "type": "string",
                                "description": "Optional crawling strategy for research mode: pseudo_adaptive (keyword-based scoring), llm_guided (LLM selects links), bfs_deep (breadth-first), research_filter (URL filtering). Only used when research_mode is true.",
                                "default": None,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

        self.crawl_counter = 0
        self.content_counter = 0
        self.total_urls = 0

    def _configure(self):
        """Validates valve configuration and logs warnings for common issues.
        Does not modify any valve values — misconfiguration must be fixed by the user.
        """
        in_docker = os.path.exists("/.dockerenv")
        if in_docker and self.valves.DEBUG:
            logger.info("Running in Docker environment")

        self._validate_url(self.valves.CRAWL4AI_BASE_URL, "CRAWL4AI_BASE_URL")
        self._validate_url(self.valves.SEARXNG_BASE_URL, "SEARXNG_BASE_URL")
        self._validate_url(self.valves.LLM_BASE_URL, "LLM_BASE_URL")

        self._validate_llm_provider()

        logger.info("Web Search and Crawl tool initialized with:")
        logger.info(f"  - Crawl4AI URL: {self.valves.CRAWL4AI_BASE_URL}")
        logger.info(f"  - LLM Provider: {self.valves.LLM_PROVIDER}")
        logger.info(f"  - LLM Base URL: {self.valves.LLM_BASE_URL}")
        logger.info(f"  - Native Search: {self.valves.USE_NATIVE_SEARCH}")
        logger.info(f"  - SearXNG: {self.valves.SEARCH_WITH_SEARXNG}")

    def _validate_url(self, url: str, name: str) -> None:
        """Warn if a valve URL is missing a protocol prefix. Does not modify the value."""
        if url and not url.startswith(("http://", "https://")):
            logger.warning(
                f"{name} is missing a protocol prefix (http:// or https://): '{url}'"
            )

    def _validate_llm_provider(self):
        """Warn if LLM provider format looks incorrect."""
        provider = self.valves.LLM_PROVIDER

        if not provider:
            logger.warning("LLM_PROVIDER is not set.")
            return

        valid_prefixes = [
            "ollama/",
            "openai/",
            "openrouter/",
            "anthropic/",
            "azure/",
            "groq/",
            "cohere/",
        ]

        if any(provider.startswith(p) for p in valid_prefixes):
            return

        if (
            "11434" in self.valves.LLM_BASE_URL
            or "ollama" in self.valves.LLM_BASE_URL.lower()
        ):
            logger.warning(
                f"LLM_PROVIDER '{provider}' looks like an Ollama model but is missing "
                f"the 'ollama/' prefix. Expected format: 'ollama/{provider}'"
            )
        else:
            logger.warning(
                f"LLM_PROVIDER '{provider}' may be missing a provider prefix. "
                f"Expected format: provider/model (e.g. openai/gpt-4o)"
            )

    # endregion

    # region ── Content Helpers ────────────────────────────────────────────────

    def _normalize_content(self, content_items: List[Any]) -> List[dict]:
        """
        Normalize extracted content to a consistent dictionary format with 'topic' and 'summary' keys.
        Handles various input shapes: dicts, lists, strings, nested structures.
        This is essential for consistent token counting and downstream processing.
        """
        normalized = []
        for item in content_items:
            if isinstance(item, dict):
                topic = item.get("topic", item.get("title", "Content"))
                summary = item.get("summary", item.get("content", ""))

                # Recursively flatten nested summaries (e.g., list of dicts)
                if isinstance(summary, list):
                    summary_texts = []
                    for s in summary:
                        if isinstance(s, dict):
                            sub_summary = s.get("summary", s.get("content", str(s)))
                            if isinstance(sub_summary, list):
                                for sub in sub_summary:
                                    if isinstance(sub, dict):
                                        summary_texts.append(
                                            sub.get("summary", str(sub))
                                        )
                                    else:
                                        summary_texts.append(str(sub))
                            else:
                                summary_texts.append(str(sub_summary))
                        else:
                            summary_texts.append(str(s))
                    summary = " ".join(summary_texts)
                elif isinstance(summary, dict):
                    summary = summary.get(
                        "summary", summary.get("content", str(summary))
                    )
                else:
                    summary = str(summary)

                normalized.append({"topic": str(topic), "summary": summary})
            elif isinstance(item, str):
                normalized.append({"topic": "Extracted information", "summary": item})
            elif isinstance(item, list):
                # Recursively process list items
                normalized.extend(self._normalize_content(item))
            else:
                normalized.append({"topic": "Content", "summary": str(item)})
        return normalized

    async def _count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))

    async def _truncate_content(
        self, content: str, max_tokens: int, model: str = "gpt-4"
    ) -> str:
        """
        Truncate content to fit within max_tokens by counting tokens with tiktoken
        and cutting at the token limit. Adds a truncation notice.
        """
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        tokens = encoding.encode(content)
        if len(tokens) <= max_tokens:
            return content

        truncated_tokens = tokens[:max_tokens]
        truncated_text = encoding.decode(truncated_tokens)
        return truncated_text + "\n\n[Content truncated due to length...]"

    # endregion

    # region ── Image Validation ───────────────────────────────────────────────

    async def _validate_image_url(self, url: str) -> bool:
        """
        Validate if an image URL is accessible and returns an image.
        Uses a HEAD request with a short timeout and checks Content-Type.
        The skip_auto_headers prevents aiohttp from adding its own Accept-Encoding,
        which could cause issues with some servers.
        """
        try:
            if not self.valves.CRAWL4AI_VALIDATE_IMAGES:
                return True

            timeout = aiohttp.ClientTimeout(total=4)
            url = url.strip()
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            async with aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                skip_auto_headers={"Accept-Encoding", "Content-Type"},
            ) as session:
                async with session.head(url, allow_redirects=True) as response:
                    if response.status != 200:
                        logger.warning(
                            f"Image validation failed for {url}: Status {response.status}"
                        )
                        return False
                    content_type = response.headers.get("Content-Type", "").lower()
                    if not content_type.startswith("image/"):
                        logger.warning(
                            f"Image validation failed for {url}: Content-Type {content_type}"
                        )
                        return False
                    return True
        except asyncio.TimeoutError:
            logger.warning(f"Image validation timeout for {url}")
            return False
        except Exception as e:
            logger.warning(f"Image validation error for {url}: {str(e)}")
            return False

    async def _validate_images_batch(self, urls: List[str]) -> List[str]:
        """Validate multiple image URLs concurrently. Returns only valid URLs."""
        tasks = [self._validate_image_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        valid_urls = [url for url, is_valid in zip(urls, results) if is_valid]
        if len(valid_urls) < len(urls) and self.valves.DEBUG:
            logger.info(
                f"Image validation: {len(valid_urls)}/{len(urls)} images are valid"
            )
        return valid_urls

    # endregion

    # region ── Search ─────────────────────────────────────────────────────────

    async def get_request(self) -> "Request":
        """Helper to create a request object for native search."""
        if not NATIVE_SEARCH_AVAILABLE:
            raise ImportError("OpenWebUI native search not available")
        return Request(scope={"type": "http", "app": app})

    async def _search_native(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
        __user__: Optional[dict] = None,
    ) -> List[str]:
        """Search using OpenWebUI's native web search and return URLs."""
        if not self.valves.USE_NATIVE_SEARCH:
            if self.valves.DEBUG:
                logger.info("Native search is disabled.")
            return []

        if not NATIVE_SEARCH_AVAILABLE:
            logger.warning("Native search not available - missing OpenWebUI imports")
            return []

        if __user__ is None:
            logger.error("User information required for native search")
            return []

        try:
            user = Users.get_user_by_id(__user__["id"])
            if user is None:
                logger.error("User not found")
                return []

            if __event_emitter__ and self.valves.MORE_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Searching using Open WebUI native search...",
                            "done": False,
                        },
                    }
                )

            form = SearchForm.model_validate({"queries": [query]})
            result = await process_web_search(
                request=Request(scope={"type": "http", "app": app}),
                form_data=form,
                user=user,
            )
            if self.valves.DEBUG:
                logger.info(f"Native search for '{query}' returned {result}")

            urls = [
                item.get("link") for item in result.get("items", []) if item.get("link")
            ]

            if self.valves.DEBUG:
                logger.info(f"Native search for '{query}' returned {len(urls)} URLs")

            if __event_emitter__ and self.valves.MORE_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Found {len(urls)} websites...",
                            "done": False,
                        },
                    }
                )

            return urls

        except Exception as e:
            logger.error(f"Error in native search: {str(e)}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Native search encountered an error: {str(e)}",
                            "done": False,
                        },
                    }
                )
            return []

    async def _search_searxng(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> List[str]:
        """Search SearXNG and return a list of URLs."""
        if not self.valves.SEARCH_WITH_SEARXNG and self.valves.DEBUG:
            logger.info("SearXNG search is disabled.")
            return []

        if not self.valves.SEARXNG_BASE_URL:
            logger.error("SearXNG base URL is not configured.")
            return []

        url = self.valves.SEARXNG_BASE_URL.replace("<query>", query)
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        if self.valves.SEARXNG_API_TOKEN:
            headers["Authorization"] = f"Bearer {self.valves.SEARXNG_API_TOKEN}"

        if __event_emitter__ and self.valves.MORE_STATUS:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": "Searching using SearXNG...",
                        "done": False,
                    },
                }
            )

        try:
            if self.valves.SEARXNG_METHOD == "POST":
                response = requests.post(
                    url,
                    data={"q": query, "format": "json"},
                    headers=headers,
                    timeout=self.valves.SEARXNG_TIMEOUT,
                )
            else:
                response = requests.get(
                    url, headers=headers, timeout=self.valves.SEARXNG_TIMEOUT
                )

            response.raise_for_status()
            data = response.json()

            results = data.get("results", [])
            max_results = (
                self.user_valves.SEARXNG_MAX_RESULTS or self.valves.SEARXNG_MAX_RESULTS
            )
            urls = [r["url"] for r in results[:max_results] if r.get("url")]

            if self.valves.DEBUG:
                logger.info(f"SearXNG search for '{query}' returned {len(urls)} URLs")

            if __event_emitter__ and self.valves.MORE_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Found {len(urls)} results...",
                            "done": False,
                        },
                    }
                )

            return urls

        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching SearXNG: {str(e)}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"SearXNG search error: {str(e)}",
                            "done": False,
                        },
                    }
                )
            return []
        except Exception as e:
            logger.error(f"Unexpected error in SearXNG search: {str(e)}")
            return []

    # endregion

    # region ── Main Entry Point ───────────────────────────────────────────────

    async def search_and_crawl(
        self,
        query: str,
        urls: Optional[List[str]] = None,
        max_results: Optional[int] = None,
        max_images: Optional[int] = None,
        research_mode: Optional[bool] = False,
        research_crawl_mode: Optional[str] = None,
        __event_emitter__: Callable[[dict], Any] = None,
        __user__: Optional[dict] = None,
    ) -> Union[list, str]:
        """
        USE THIS TOOL whenever the user asks to 'search' for, 'lookup', 'find' information,
        'browse' the web, 'gather' data on a specific topic, or when any information or data
        is needed from the internet to respond to the user.

        This tool performs web searches using both Native Search and/or SearXNG to gather
        relevant URLs, then crawls those URLs using Crawl4AI to extract clean content with media.

        :param query: The search query to use.
        :param urls: Optional list of URLs to crawl in addition to those found from searching.
        :param max_results: The maximum number of search results to crawl (per search).
        :param max_images: The maximum number of images results to display in the chat window.
        :param research_mode: Enables Research Mode for deeper web crawling with advanced strategies.
        :param research_crawl_mode: Optional crawling strategy for research mode:
            - pseudo_adaptive: Keyword-based URL scoring and iterative crawling
            - llm_guided: Use LLM to intelligently select which links to crawl next
            - bfs_deep: Breadth-first search style deep crawling
            - research_filter: Research mode with URL filtering and relevance scoring
        """
        logger.info(f"Starting search and crawl for '{query}'")

        gathered_urls = []
        self.crawl_counter = 0
        self.content_counter = 0
        self.total_urls = 0

        if not max_images:
            max_images = (
                self.user_valves.CRAWL4AI_MAX_MEDIA_ITEMS
                or self.valves.CRAWL4AI_MAX_MEDIA_ITEMS
            )

        if urls:
            for url in urls:
                if not url.startswith("http"):
                    url = f"https://{url}"
                if url not in gathered_urls:
                    gathered_urls.append(url)

        if __event_emitter__ and str(self.valves.INITIAL_RESPONSE).strip() != "":
            await __event_emitter__(
                {
                    "type": "chat:message:delta",
                    "data": {"content": str(self.valves.INITIAL_RESPONSE).strip()},
                }
            )

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Searching for '{query}'...",
                        "done": False,
                    },
                }
            )

        if self.valves.USE_NATIVE_SEARCH:
            native_urls = await self._search_native(query, __event_emitter__, __user__)
            for url in native_urls:
                if url not in gathered_urls:
                    gathered_urls.append(url)

        if self.valves.SEARCH_WITH_SEARXNG:
            searxng_urls = await self._search_searxng(query, __event_emitter__)
            max_results = (
                self.user_valves.SEARXNG_MAX_RESULTS
                or max_results
                or self.valves.SEARXNG_MAX_RESULTS
            )
            for url in searxng_urls[:max_results]:
                if url not in gathered_urls:
                    gathered_urls.append(url)

        if not gathered_urls:
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Nothing found for query '{query}'.",
                            "done": True,
                        },
                    }
                )
            if self.valves.DEBUG:
                logger.info(f"No URLs gathered to crawl for query '{query}'.")
            return f"No URLs found to crawl for the query: {query}."

        max_urls = self.user_valves.CRAWL4AI_MAX_URLS or self.valves.CRAWL4AI_MAX_URLS
        if len(gathered_urls) > max_urls:
            gathered_urls = gathered_urls[:max_urls]

        if __event_emitter__ and self.valves.MORE_STATUS:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Found {len(gathered_urls)} results. Inspecting the content...",
                        "done": False,
                    },
                }
            )

        effective_research_mode = research_mode or self.user_valves.RESEARCH_MODE
        effective_crawl_mode = (
            research_crawl_mode or self.user_valves.RESEARCH_CRAWL_MODE
        )

        crawl_results = []
        batch_count = 1
        image_list = []
        video_list = []
        seen_images = set()
        seen_videos = set()
        total_tokens = 0
        thumbnail_size = (
            self.user_valves.CRAWL4AI_THUMBNAIL_SIZE
            or self.valves.CRAWL4AI_THUMBNAIL_SIZE
            or 200
        )
        self.total_urls = len(gathered_urls)

        # ── Research mode ──────────────────────────────────────────────────────
        if effective_research_mode and len(gathered_urls) > 0:
            if __event_emitter__ and self.valves.MORE_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Research Mode enabled. Using '{effective_crawl_mode}' strategy...",
                            "done": False,
                        },
                    }
                )

            research_result = await self._research_crawl(
                urls=gathered_urls,
                query=query,
                mode=effective_crawl_mode,
                max_tokens=self.valves.CRAWL4AI_MAX_TOKENS,  # Token limit passed
                __event_emitter__=__event_emitter__,
            )

            if "content" in research_result:
                # Already normalized inside research methods
                crawl_results.extend(research_result["content"])
                if self.valves.DEBUG:
                    logger.info(
                        f"Research mode added {len(research_result['content'])} content items"
                    )
            if "images" in research_result:
                image_list.extend(research_result["images"])
            if "videos" in research_result:
                video_list.extend(research_result["videos"])

        # ── Standard batch crawl ───────────────────────────────────────────────
        else:
            for i in range(0, len(gathered_urls), self.valves.CRAWL4AI_BATCH):
                batch = gathered_urls[i : i + self.valves.CRAWL4AI_BATCH]
                try:
                    crawled_batch = await self._crawl_url(
                        urls=batch, query=query, __event_emitter__=__event_emitter__
                    )

                    if self.valves.DEBUG:
                        logger.info(
                            f"Found {len(crawled_batch.get('content', []))} content, "
                            f"{len(crawled_batch.get('images', []))} images, "
                            f"{len(crawled_batch.get('videos', []))} videos."
                        )

                    # Compile images (deduplicate by normalized base URL)
                    for img_url in crawled_batch.get("images", []):
                        parsed_image = urlparse(img_url)
                        base_image_url = f"{parsed_image.scheme}://{parsed_image.netloc}{parsed_image.path}"
                        if base_image_url in seen_images:
                            continue
                        seen_images.add(base_image_url)
                        # Create a thumbnail URL using images.weserv.nl service
                        thumbnail_url = (
                            f"https://images.weserv.nl/?url={quote(img_url)}"
                            f"&w={thumbnail_size}&h={thumbnail_size}&fit=inside"
                        )
                        # Validate both the original and the thumbnail URL
                        if await self._validate_image_url(
                            img_url
                        ) and await self._validate_image_url(thumbnail_url):
                            image_list.append(img_url)

                    # Compile videos (deduplicate by normalized base URL)
                    for vid_url in crawled_batch.get("videos", []):
                        parsed_video = urlparse(vid_url)
                        base_video_url = f"{parsed_video.scheme}://{parsed_video.netloc}{parsed_video.path}"
                        if base_video_url in seen_videos:
                            continue
                        seen_videos.add(base_video_url)
                        video_list.append(vid_url)

                    # Process content with optional token limit
                    data_list = crawled_batch.get("content", [])
                    normalized_data_list = self._normalize_content(data_list)

                    if normalized_data_list:
                        # Serialize to JSON to count tokens (tiktoken counts tokens in the actual string)
                        content_str = orjson.dumps(normalized_data_list).decode("utf-8")
                        page_tokens = await self._count_tokens(content_str)

                        # Apply per‑page token limit if configured
                        if (
                            self.valves.CRAWL4AI_MAX_TOKENS > 0
                            and page_tokens > self.valves.CRAWL4AI_MAX_TOKENS
                        ):
                            content_str = await self._truncate_content(
                                content_str, self.valves.CRAWL4AI_MAX_TOKENS
                            )
                            # Re‑parse truncated JSON to keep structure
                            try:
                                normalized_data_list = orjson.loads(
                                    content_str.replace(
                                        "\n\n[Content truncated due to length...]", ""
                                    )
                                )
                            except Exception:
                                pass
                            page_tokens = self.valves.CRAWL4AI_MAX_TOKENS
                            if self.valves.DEBUG:
                                logger.info(
                                    f"Truncated content from batch to {self.valves.CRAWL4AI_MAX_TOKENS} tokens"
                                )

                        # Check global token budget (sum across all pages)
                        if (
                            self.valves.CRAWL4AI_MAX_TOKENS > 0
                            and total_tokens + page_tokens
                            > self.valves.CRAWL4AI_MAX_TOKENS
                        ):
                            logger.warning(
                                f"Reached token limit ({self.valves.CRAWL4AI_MAX_TOKENS}). Skipping remaining pages."
                            )
                            if __event_emitter__ and self.valves.MORE_STATUS:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": f"Token limit reached. Processed {len(crawl_results)} pages.",
                                            "done": False,
                                        },
                                    }
                                )
                            continue

                        total_tokens += page_tokens
                        if self.valves.DEBUG:
                            limit_label = (
                                self.valves.CRAWL4AI_MAX_TOKENS
                                if self.valves.CRAWL4AI_MAX_TOKENS > 0
                                else "unlimited"
                            )
                            logger.info(
                                f"Batch {batch_count}: {page_tokens} tokens (Total: {total_tokens}/{limit_label})"
                            )

                        crawl_results.extend(normalized_data_list)

                    batch_count += 1

                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
                    )

        # Final normalization pass (for safety)
        crawl_results = self._normalize_content(crawl_results)

        if self.valves.DEBUG:
            logger.info(f"Final crawl_results count: {len(crawl_results)}")
            for idx, item in enumerate(crawl_results[:3]):
                logger.info(f"Sample {idx}: {type(item)} - {str(item)[:100]}")

        # ── Display media ──────────────────────────────────────────────────────
        if __event_emitter__ and (
            self.user_valves.CRAWL4AI_DISPLAY_MEDIA
            or self.valves.CRAWL4AI_DISPLAY_MEDIA
        ):
            max_items = self.valves.CRAWL4AI_MAX_MEDIA_ITEMS
            image_list = image_list[:max_images] if max_images > 0 else image_list
            video_list = video_list[:max_items] if max_items > 0 else video_list

            if image_list:
                image_markdown = ""
                for img_url in image_list:
                    if (
                        self.user_valves.CRAWL4AI_DISPLAY_THUMBNAILS
                        or self.valves.CRAWL4AI_DISPLAY_THUMBNAILS
                    ):
                        thumbnail_url = (
                            f"https://images.weserv.nl/?url={quote(img_url)}"
                            f"&w={thumbnail_size}&h={thumbnail_size}&fit=inside"
                        )
                    else:
                        thumbnail_url = img_url
                    image_markdown += f"[![image]({thumbnail_url})]({img_url})\n"
                await __event_emitter__(
                    {"type": "message", "data": {"content": image_markdown}}
                )

            if video_list:
                video_markdown = "\n\n*Videos links:*\n"
                for idx, vid_url in enumerate(video_list, 1):
                    video_markdown += f"{idx}. [{vid_url}]({vid_url})\n"
                await __event_emitter__(
                    {"type": "message", "data": {"content": video_markdown}}
                )

        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Inspected {len(crawl_results)} web pages.",
                        "done": True,
                    },
                }
            )

        return crawl_results

    # endregion

    # region ── Research Crawl Router ──────────────────────────────────────────

    async def _research_crawl(
        self,
        urls: List[str],
        query: str,
        mode: str = "pseudo_adaptive",
        max_tokens: int = 0,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """Route to the appropriate research crawling strategy."""
        # Each strategy has a different approach:
        # - pseudo_adaptive: keyword scoring priority queue
        # - llm_guided: LLM selects next links (placeholder for now)
        # - bfs_deep: breadth‑first search with depth limit
        # - research_filter: seed URLs + follow high‑scoring links
        if mode == ResearchCrawlMode.PSEUDO_ADAPTIVE:
            return await self._pseudo_adaptive_crawl(
                urls, query, max_tokens, __event_emitter__
            )
        elif mode == ResearchCrawlMode.LLM_GUIDED:
            return await self._llm_guided_crawl(
                urls, query, max_tokens, __event_emitter__
            )
        elif mode == ResearchCrawlMode.BFS_DEEP:
            return await self._bfs_deep_crawl(
                urls, query, max_tokens, __event_emitter__
            )
        elif mode == ResearchCrawlMode.RESEARCH_FILTER:
            return await self._research_filter_crawl(
                urls, query, max_tokens, __event_emitter__
            )
        else:
            logger.warning(
                f"Unknown research crawl mode: {mode}, defaulting to pseudo_adaptive"
            )
            return await self._pseudo_adaptive_crawl(
                urls, query, max_tokens, __event_emitter__
            )

    # endregion

    # region ── Research Crawl Strategies ─────────────────────────────────────

    async def _pseudo_adaptive_crawl(
        self,
        start_urls: List[str],
        query: str,
        max_tokens: int = 0,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """
        Pseudo‑adaptive crawl: scores URLs based on keyword match, uses a priority queue.
        It crawls the highest‑scoring URLs first up to max_pages and max_depth.
        """
        from collections import deque

        max_pages = self.user_valves.RESEARCH_MAX_PAGES
        max_depth = self.user_valves.RESEARCH_MAX_DEPTH
        batch_size = self.user_valves.RESEARCH_BATCH_SIZE
        include_external = self.user_valves.RESEARCH_INCLUDE_EXTERNAL

        keywords = query.lower().split()
        crawled_pages = set()
        crawled_results = []
        all_images = []
        all_videos = []
        total_tokens = 0

        # Initial queue: each entry is (url, depth, score)
        queue = deque()
        for url in start_urls[:5]:
            if url not in crawled_pages:
                score = sum(1 for kw in keywords if kw in url.lower())
                queue.append((url, 0, score))

        self.total_urls = max_pages

        while (
            queue
            and len(crawled_pages) < max_pages
            and (max_tokens == 0 or total_tokens < max_tokens)
        ):
            # Take a batch from the queue, sort by score descending
            batch = []
            for _ in range(min(batch_size, len(queue))):
                if queue:
                    batch.append(queue.popleft())

            batch.sort(key=lambda x: x[2], reverse=True)

            for url, depth, score in batch:
                if len(crawled_pages) >= max_pages or depth > max_depth:
                    continue
                if url in crawled_pages:
                    continue

                crawled_pages.add(url)

                if __event_emitter__ and self.valves.MORE_STATUS:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[Pseudo-Adaptive] Depth {depth}: Crawling {url[:60]}... ({len(crawled_pages)}/{max_pages})",
                                "done": False,
                            },
                        }
                    )

                result = await self._crawl_url(
                    urls=[url],
                    query=query,
                    extract_links=True,  # Request link extraction for deeper crawling.
                    __event_emitter__=__event_emitter__,
                )

                if result.get("content"):
                    normalized_content = self._normalize_content(result["content"])
                    if normalized_content:
                        content_str = orjson.dumps(normalized_content).decode("utf-8")
                        page_tokens = await self._count_tokens(content_str)

                        # Apply per‑page token truncation if needed
                        if max_tokens > 0 and page_tokens > max_tokens:
                            content_str = await self._truncate_content(
                                content_str, max_tokens
                            )
                            try:
                                normalized_content = orjson.loads(
                                    content_str.replace(
                                        "\n\n[Content truncated due to length...]", ""
                                    )
                                )
                            except Exception:
                                pass
                            page_tokens = max_tokens
                            if self.valves.DEBUG:
                                logger.info(
                                    f"Truncated content from {url} to {max_tokens} tokens"
                                )

                        # Check global token budget
                        if max_tokens > 0 and total_tokens + page_tokens > max_tokens:
                            logger.warning(
                                f"Token limit reached. Stopping further research crawling."
                            )
                            if __event_emitter__ and self.valves.MORE_STATUS:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": f"Token limit reached. Processed {len(crawled_results)} content items.",
                                            "done": False,
                                        },
                                    }
                                )
                            break
                        else:
                            total_tokens += page_tokens
                            crawled_results.extend(normalized_content)

                if result.get("images"):
                    all_images.extend(result["images"])
                if result.get("videos"):
                    all_videos.extend(result["videos"])

                if max_tokens > 0 and total_tokens >= max_tokens:
                    break

                # Enqueue new links if depth not exceeded
                if depth < max_depth:
                    for link in result.get("links", []):
                        if link in crawled_pages:
                            continue
                        parsed_link = urlparse(link)
                        parsed_url = urlparse(url)
                        # Respect domain restrictions
                        if (
                            not include_external
                            and parsed_link.netloc
                            and parsed_link.netloc != parsed_url.netloc
                        ):
                            continue
                        link_score = sum(1 for kw in keywords if kw in link.lower())
                        if link_score > 0:
                            queue.append((link, depth + 1, link_score))

            if max_tokens > 0 and total_tokens >= max_tokens:
                break

        if self.valves.DEBUG:
            logger.info(
                f"[Pseudo-Adaptive] Crawled {len(crawled_pages)} pages, used {total_tokens} tokens"
            )

        return {
            "content": crawled_results,
            "images": all_images,
            "videos": all_videos,
            "pages_crawled": len(crawled_pages),
            "tokens_used": total_tokens,
        }

    async def _llm_guided_crawl(
        self,
        start_urls: List[str],
        query: str,
        max_tokens: int = 0,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """
        LLM‑guided crawl: uses an LLM to select which links to follow next.
        (Currently only scores links by keyword; full LLM selection is a placeholder.)
        """
        max_pages = self.user_valves.RESEARCH_MAX_PAGES
        include_external = self.user_valves.RESEARCH_INCLUDE_EXTERNAL

        crawled_pages = set()
        crawled_results = []
        all_images = []
        all_videos = []
        total_tokens = 0

        llm_config = LLMConfig(
            provider=self.valves.LLM_PROVIDER,
            base_url=self.valves.LLM_BASE_URL.rstrip("/"),
            temperature=0.3,
            max_tokens=500,
        )
        llm_config.api_token = (
            self.valves.LLM_API_TOKEN if self.valves.LLM_API_TOKEN else None
        )

        urls_to_process = list(start_urls[:5])

        while (
            urls_to_process
            and len(crawled_pages) < max_pages
            and (max_tokens == 0 or total_tokens < max_tokens)
        ):
            current_url = urls_to_process.pop(0)

            if current_url in crawled_pages:
                continue

            crawled_pages.add(current_url)

            if __event_emitter__ and self.valves.MORE_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"[LLM-Guided] Crawling {current_url[:60]}... ({len(crawled_pages)}/{max_pages})",
                            "done": False,
                        },
                    }
                )

            result = await self._crawl_url(
                urls=[current_url],
                query=query,
                extract_links=True,  # Request link extraction for deeper crawling.
                __event_emitter__=__event_emitter__,
            )

            if result.get("content"):
                normalized_content = self._normalize_content(result["content"])
                if normalized_content:
                    content_str = orjson.dumps(normalized_content).decode("utf-8")
                    page_tokens = await self._count_tokens(content_str)

                    if max_tokens > 0 and page_tokens > max_tokens:
                        content_str = await self._truncate_content(
                            content_str, max_tokens
                        )
                        try:
                            normalized_content = orjson.loads(
                                content_str.replace(
                                    "\n\n[Content truncated due to length...]", ""
                                )
                            )
                        except Exception:
                            pass
                        page_tokens = max_tokens
                        if self.valves.DEBUG:
                            logger.info(
                                f"Truncated content from {current_url} to {max_tokens} tokens"
                            )

                    if max_tokens > 0 and total_tokens + page_tokens > max_tokens:
                        logger.warning(
                            f"Token limit reached. Stopping further research crawling."
                        )
                        if __event_emitter__ and self.valves.MORE_STATUS:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"Token limit reached. Processed {len(crawled_results)} content items.",
                                        "done": False,
                                    },
                                }
                            )
                        break
                    else:
                        total_tokens += page_tokens
                        crawled_results.extend(normalized_content)

            if result.get("images"):
                all_images.extend(result["images"])
            if result.get("videos"):
                all_videos.extend(result["videos"])

            if max_tokens > 0 and total_tokens >= max_tokens:
                break

            discovered_links = result.get("links", [])[:15]
            if not discovered_links:
                continue

            if not include_external:
                parsed_current = urlparse(current_url)
                discovered_links = [
                    link
                    for link in discovered_links
                    if not urlparse(link).netloc
                    or urlparse(link).netloc == parsed_current.netloc
                ]

            if not discovered_links:
                continue

            # Currently falls back to keyword scoring.
            keywords = query.lower().split()
            scored_links = []
            for link in discovered_links:
                if link in crawled_pages or link in urls_to_process:
                    continue
                score = sum(1 for kw in keywords if kw in link.lower())
                if score > 0:
                    scored_links.append((link, score))
            scored_links.sort(key=lambda x: x[1], reverse=True)

            for link, _ in scored_links[:3]:
                if link not in urls_to_process and link not in crawled_pages:
                    urls_to_process.append(link)

            if max_tokens > 0 and total_tokens >= max_tokens:
                break

        if self.valves.DEBUG:
            logger.info(
                f"[LLM-Guided] Crawled {len(crawled_pages)} pages, used {total_tokens} tokens"
            )

        return {
            "content": crawled_results,
            "images": all_images,
            "videos": all_videos,
            "pages_crawled": len(crawled_pages),
            "tokens_used": total_tokens,
        }

    async def _bfs_deep_crawl(
        self,
        start_urls: List[str],
        query: str,
        max_tokens: int = 0,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """
        Breadth‑first deep crawl: explores pages layer by layer up to max_depth,
        respecting domain restrictions.
        """
        from collections import deque

        max_pages = self.user_valves.RESEARCH_MAX_PAGES
        max_depth = self.user_valves.RESEARCH_MAX_DEPTH
        batch_size = self.user_valves.RESEARCH_BATCH_SIZE
        include_external = self.user_valves.RESEARCH_INCLUDE_EXTERNAL

        crawled_pages = set()
        crawled_results = []
        all_images = []
        all_videos = []
        total_tokens = 0

        base_domain = urlparse(start_urls[0]).netloc if start_urls else ""

        queue = deque((url, 0) for url in start_urls[:5] if url not in crawled_pages)
        self.total_urls = max_pages

        while (
            queue
            and len(crawled_pages) < max_pages
            and (max_tokens == 0 or total_tokens < max_tokens)
        ):
            level_batch = [queue.popleft() for _ in range(min(batch_size, len(queue)))]

            for url, depth in level_batch:
                if (
                    len(crawled_pages) >= max_pages
                    or depth > max_depth
                    or url in crawled_pages
                ):
                    continue

                crawled_pages.add(url)

                if __event_emitter__ and self.valves.MORE_STATUS:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[BFS-Deep] Depth {depth}: Crawling {url[:60]}... ({len(crawled_pages)}/{max_pages})",
                                "done": False,
                            },
                        }
                    )

                result = await self._crawl_url(
                    urls=[url],
                    query=query,
                    extract_links=True,  # Request link extraction for deeper crawling.
                    __event_emitter__=__event_emitter__,
                )

                if result.get("content"):
                    normalized_content = self._normalize_content(result["content"])
                    if normalized_content:
                        content_str = orjson.dumps(normalized_content).decode("utf-8")
                        page_tokens = await self._count_tokens(content_str)

                        if max_tokens > 0 and page_tokens > max_tokens:
                            content_str = await self._truncate_content(
                                content_str, max_tokens
                            )
                            try:
                                normalized_content = orjson.loads(
                                    content_str.replace(
                                        "\n\n[Content truncated due to length...]", ""
                                    )
                                )
                            except Exception:
                                pass
                            page_tokens = max_tokens
                            if self.valves.DEBUG:
                                logger.info(
                                    f"Truncated content from {url} to {max_tokens} tokens"
                                )

                        if max_tokens > 0 and total_tokens + page_tokens > max_tokens:
                            logger.warning(
                                f"Token limit reached. Stopping further research crawling."
                            )
                            if __event_emitter__ and self.valves.MORE_STATUS:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": f"Token limit reached. Processed {len(crawled_results)} content items.",
                                            "done": False,
                                        },
                                    }
                                )
                            break
                        else:
                            total_tokens += page_tokens
                            crawled_results.extend(normalized_content)

                if result.get("images"):
                    all_images.extend(result["images"])
                if result.get("videos"):
                    all_videos.extend(result["videos"])

                if max_tokens > 0 and total_tokens >= max_tokens:
                    break

                if depth < max_depth:
                    for link in result.get("links", [])[:10]:
                        if link in crawled_pages:
                            continue
                        parsed_link = urlparse(link)
                        if (
                            not include_external
                            and parsed_link.netloc
                            and parsed_link.netloc != base_domain
                        ):
                            continue
                        queue.append((link, depth + 1))

            if max_tokens > 0 and total_tokens >= max_tokens:
                break

        if self.valves.DEBUG:
            logger.info(
                f"[BFS-Deep] Crawled {len(crawled_pages)} pages, used {total_tokens} tokens"
            )

        return {
            "content": crawled_results,
            "images": all_images,
            "videos": all_videos,
            "pages_crawled": len(crawled_pages),
            "tokens_used": total_tokens,
        }

    async def _research_filter_crawl(
        self,
        start_urls: List[str],
        query: str,
        max_tokens: int = 0,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """
        Research‑filter crawl: starts from seed URLs, extracts content and relevant links,
        then follows the most promising links based on keyword score.
        """
        max_pages = self.user_valves.RESEARCH_MAX_PAGES
        include_external = self.user_valves.RESEARCH_INCLUDE_EXTERNAL
        keywords = query.lower().split()

        results = {
            "content": [],
            "images": [],
            "videos": [],
            "sources": {},
            "total_pages": 0,
            "tokens_used": 0,
        }

        total_tokens = 0

        for source_url in start_urls[:5]:  # Max 5 starting sources
            # Stop if we already reached the page limit
            if results["total_pages"] >= max_pages:
                break

            if __event_emitter__ and self.valves.MORE_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"[Research-Filter] Researching: {source_url[:60]}... ({results['total_pages']}/{max_pages})",
                            "done": False,
                        },
                    }
                )

            source_result = await self._crawl_url(
                urls=[source_url],
                query=query,
                extract_links=True,  # Request link extraction for deeper crawling.
                __event_emitter__=__event_emitter__,
            )

            if source_result.get("content"):
                normalized_content = self._normalize_content(source_result["content"])
                if normalized_content:
                    content_str = orjson.dumps(normalized_content).decode("utf-8")
                    page_tokens = await self._count_tokens(content_str)

                    if max_tokens > 0 and page_tokens > max_tokens:
                        content_str = await self._truncate_content(
                            content_str, max_tokens
                        )
                        try:
                            normalized_content = orjson.loads(
                                content_str.replace(
                                    "\n\n[Content truncated due to length...]", ""
                                )
                            )
                        except Exception:
                            pass
                        page_tokens = max_tokens
                        if self.valves.DEBUG:
                            logger.info(
                                f"Truncated content from {source_url} to {max_tokens} tokens"
                            )

                    if max_tokens > 0 and total_tokens + page_tokens > max_tokens:
                        logger.warning(
                            f"Token limit reached. Stopping further research crawling."
                        )
                        if __event_emitter__ and self.valves.MORE_STATUS:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"Token limit reached. Processed {results['total_pages']} pages.",
                                        "done": False,
                                    },
                                }
                            )
                        break
                    else:
                        total_tokens += page_tokens
                        relevance_score = sum(
                            1
                            for kw in keywords
                            if kw in str(normalized_content).lower()
                        )
                        results["sources"][source_url] = {
                            "content": normalized_content,
                            "relevance_score": relevance_score,
                            "links": source_result.get("links", [])[:10],
                        }
                        results["content"].extend(normalized_content)
                        results["total_pages"] += 1

            results["images"].extend(source_result.get("images", []))
            results["videos"].extend(source_result.get("videos", []))

            if max_tokens > 0 and total_tokens >= max_tokens:
                break

            # Score links from the source page
            scored_links = []
            for link in source_result.get("links", [])[:15]:
                if results["total_pages"] >= max_pages:
                    break
                score = sum(1 for kw in keywords if kw in link.lower())
                if score > 0:
                    scored_links.append((link, score))

            # Sort links by relevance to crawl the most promising first
            scored_links.sort(key=lambda x: x[1], reverse=True)

            crawled = 0
            for link, _ in scored_links:
                if results["total_pages"] >= max_pages or crawled >= 3:
                    break
                if max_tokens > 0 and total_tokens >= max_tokens:
                    break
                # Check domain
                if not include_external:
                    parsed_link = urlparse(link)
                    parsed_source = urlparse(source_url)
                    if (
                        parsed_link.netloc
                        and parsed_link.netloc != parsed_source.netloc
                    ):
                        continue

                if __event_emitter__ and self.valves.MORE_STATUS:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"[Research-Filter] Following: {link[:60]}...",
                                "done": False,
                            },
                        }
                    )

                link_result = await self._crawl_url(
                    urls=[link], query=query, __event_emitter__=__event_emitter__
                )

                if link_result.get("content"):
                    normalized_link_content = self._normalize_content(
                        link_result["content"]
                    )
                    if normalized_link_content:
                        content_str = orjson.dumps(normalized_link_content).decode(
                            "utf-8"
                        )
                        page_tokens = await self._count_tokens(content_str)

                        if max_tokens > 0 and page_tokens > max_tokens:
                            content_str = await self._truncate_content(
                                content_str, max_tokens
                            )
                            try:
                                normalized_link_content = orjson.loads(
                                    content_str.replace(
                                        "\n\n[Content truncated due to length...]", ""
                                    )
                                )
                            except Exception:
                                pass
                            page_tokens = max_tokens
                            if self.valves.DEBUG:
                                logger.info(
                                    f"Truncated content from {link} to {max_tokens} tokens"
                                )

                        if max_tokens > 0 and total_tokens + page_tokens > max_tokens:
                            logger.warning(
                                f"Token limit reached. Stopping further research crawling."
                            )
                            if __event_emitter__ and self.valves.MORE_STATUS:
                                await __event_emitter__(
                                    {
                                        "type": "status",
                                        "data": {
                                            "description": f"Token limit reached. Processed {results['total_pages']} pages.",
                                            "done": False,
                                        },
                                    }
                                )
                            break
                        else:
                            total_tokens += page_tokens
                            results["content"].extend(normalized_link_content)
                            results["total_pages"] += 1
                            crawled += 1

                results["images"].extend(link_result.get("images", []))
                results["videos"].extend(link_result.get("videos", []))

            if max_tokens > 0 and total_tokens >= max_tokens:
                break

        # Final normalization and sort by relevance
        results["content"] = self._normalize_content(results["content"])
        results["content"].sort(
            key=lambda x: sum(
                1 for kw in keywords if kw in x.get("summary", "").lower()
            ),
            reverse=True,
        )
        results["tokens_used"] = total_tokens

        if self.valves.DEBUG:
            logger.info(
                f"[Research-Filter] Crawled {results['total_pages']} pages, used {total_tokens} tokens"
            )

        return results

    # endregion

    # region ── Core Crawler ───────────────────────────────────────────────────

    async def _crawl_url(
        self,
        urls: Union[list, str],
        query: Optional[str] = None,
        extract_links: bool = False,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """
        Internal function to crawl URLs and extract content.
        Converts any webpage into clean content and extracts images and videos.

        :param urls: The exact web URL(s) to extract data from.
        :param query: Optional search query for relevance context.
        :param extract_links: Whether to extract and return discovered links for research mode.
        """
        if isinstance(urls, str):
            urls = [urls]

        # Normalize user-supplied URLs (runtime input, not valves)
        urls = [
            url if url.startswith(("http://", "https://")) else f"https://{url}"
            for url in urls
        ]

        base_url = self.valves.CRAWL4AI_BASE_URL.rstrip("/")
        endpoint = f"{base_url}/crawl"

        if self.valves.DEBUG:
            logger.info(f"Using LLM provider: {self.valves.LLM_PROVIDER}")
            logger.info(f"Crawl4AI endpoint: {endpoint}")
            logger.info(f"URLs to crawl: {urls}")

        browser_config = BrowserConfig(
            headless=True,
            light_mode=True,
            headers={
                "sec-ch-ua": '"Chromium";v="116", "Not_A Brand";v="8", "Google Chrome";v="116"'
            },
            extra_args=["--no-sandbox", "--disable-gpu"],
        )

        llm_config = LLMConfig(
            provider=self.valves.LLM_PROVIDER,
            base_url=self.valves.LLM_BASE_URL.rstrip("/"),
            temperature=self.valves.LLM_TEMPERATURE or 0.3,
            max_tokens=self.valves.LLM_MAX_TOKENS or None,
            top_p=self.valves.LLM_TOP_P or None,
            frequency_penalty=self.valves.LLM_FREQUENCY_PENALTY or None,
            presence_penalty=self.valves.LLM_PRESENCE_PENALTY or None,
        )
        llm_config.api_token = (
            self.valves.LLM_API_TOKEN
            if self.valves.LLM_API_TOKEN and self.valves.LLM_API_TOKEN.strip()
            else None
        )

        if "ollama" in self.valves.LLM_PROVIDER.lower():
            if self.valves.LLM_BASE_URL.startswith("https://"):
                logger.warning(
                    f"Ollama typically uses HTTP, not HTTPS. "
                    f"Your LLM_BASE_URL is set to '{self.valves.LLM_BASE_URL}'. "
                    f"Update it to http:// if you don't have a valid TLS certificate."
                )
            elif not self.valves.LLM_BASE_URL.startswith("http://"):
                logger.warning(
                    f"Ollama base URL should start with http://. "
                    f"Current LLM_BASE_URL: '{self.valves.LLM_BASE_URL}'"
                )

        extraction_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            instruction=self.valves.LLM_INSTRUCTION,
            input_format="fit_markdown",
            schema=ArticleData.model_json_schema(),
        )

        md_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(),
            options={"ignore_links": True, "escape_html": False, "body_width": 80},
        )

        crawler_config = CrawlerRunConfig(
            markdown_generator=md_generator,
            extraction_strategy=extraction_strategy,
            table_extraction=DefaultTableExtraction(),
            exclude_external_links=not self.valves.CRAWL4AI_EXTERNAL_DOMAINS,
            exclude_social_media_domains=[
                d.strip()
                for d in self.valves.CRAWL4AI_EXCLUDE_SOCIAL_MEDIA_DOMAINS.split(",")
                if d.strip()
            ],
            exclude_domains=[
                d.strip()
                for d in self.valves.CRAWL4AI_EXCLUDE_DOMAINS.split(",")
                if d.strip()
            ],
            user_agent=self.valves.CRAWL4AI_USER_AGENT,
            stream=False,
            cache_mode=CacheMode.BYPASS,
            page_timeout=self.valves.CRAWL4AI_TIMEOUT * 1000,
            only_text=self.valves.CRAWL4AI_TEXT_ONLY,
            word_count_threshold=self.valves.CRAWL4AI_WORD_COUNT_THRESHOLD,
            exclude_all_images=self.valves.CRAWL4AI_EXCLUDE_IMAGES == "All",
            exclude_external_images=self.valves.CRAWL4AI_EXCLUDE_IMAGES == "External",
        )

        if __event_emitter__ and self.valves.MORE_STATUS:
            description = (
                f"Processing {len(urls)} URLs..."
                if len(urls) > 1
                else f"Processing {urls[0]}..."
            )
            await __event_emitter__(
                {"type": "status", "data": {"description": description, "done": False}}
            )

        self.crawl_counter += len(urls)

        if self.valves.DEBUG:
            logger.info(f"Contacting Crawl4AI at {endpoint} for URLs: {urls}")

        headers = {"Content-Type": "application/json"}
        payload = {
            "urls": urls,
            "browser_config": browser_config.dump(),
            "crawler_config": crawler_config.dump(),
        }

        try:
            timeout = self.valves.CRAWL4AI_TIMEOUT * len(urls) + 60
            response = requests.post(
                endpoint, json=payload, headers=headers, timeout=timeout
            )
            response.raise_for_status()
            data = response.json()

            results = []
            seen_images = set()
            seen_videos = set()
            all_images = []
            all_videos = []
            all_links = []

            for item in data.get("results", []):
                if item.get("success") is not True:
                    continue

                url = item.get("url", "")
                parsed_url = urlparse(url)

                # ── Extract images with scoring ─────────────────────────────────
                # Only include images that meet the minimum score threshold.
                image_list = []
                for img in filter(
                    lambda x: x.get("score", 0) >= self.valves.CRAWL4AI_MIN_IMAGE_SCORE,
                    item.get("media", {}).get("images", []),
                ):
                    src = img.get("src")
                    if not src:
                        continue
                    if src.startswith("//"):
                        src = f"https:{src}"
                    elif not src.startswith("http"):
                        src = f"{parsed_url.scheme}://{parsed_url.netloc}/{src.lstrip('/')}"
                    parsed_image = urlparse(src)
                    key = f"{parsed_image.scheme}://{parsed_image.netloc}/{parsed_image.path}"
                    if key not in seen_images:
                        seen_images.add(key)
                        image_list.append(src)

                # ── Extract videos (simple heuristic, score ≥ 5) ─────────────────
                video_list = []
                for vid in filter(
                    lambda x: x.get("score", 0) >= 5,
                    item.get("media", {}).get("videos", []),
                ):
                    src = vid.get("src")
                    if not src:
                        continue
                    if src.startswith("//"):
                        src = f"https:{src}"
                    elif not src.startswith("http"):
                        src = f"{parsed_url.scheme}://{parsed_url.netloc}/{src.lstrip('/')}"
                    parsed_video = urlparse(src)
                    key = f"{parsed_video.scheme}://{parsed_video.netloc}/{parsed_video.path}"
                    if key not in seen_videos:
                        seen_videos.add(key)
                        video_list.append(src)

                # ── Extract links (research mode) ──────────────────────────────
                if extract_links:
                    html_content = item.get("html", "")
                    for match in re.findall(r'href=["\'](.*?)["\']', html_content):
                        if not match or match.startswith(("#", "javascript:")):
                            continue
                        if not match.startswith("http"):
                            match = (
                                f"{parsed_url.scheme}://{parsed_url.netloc}{match}"
                                if match.startswith("/")
                                else f"{parsed_url.scheme}://{parsed_url.netloc}/{match}"
                            )
                        if match.startswith("http"):
                            all_links.append(match)

                await __event_emitter__(
                    {"type": "files", "data": {"files": image_list + video_list}}
                )

                # ── Parse extracted_content from Crawl4AI (JSON string or object) ─
                try:
                    extracted_content = item.get("extracted_content", "[]")
                    if isinstance(extracted_content, str):
                        tmp_content = orjson.loads(extracted_content)
                    else:
                        tmp_content = extracted_content

                    if not isinstance(tmp_content, list):
                        if self.valves.DEBUG:
                            logger.warning(
                                f"extracted_content is not a list: {type(tmp_content)}"
                            )
                        tmp_content = []

                    content_list = []
                    for content_item in tmp_content:
                        if (
                            isinstance(content_item, dict)
                            and content_item.get("error") is not False
                        ):
                            content_list.append(
                                {
                                    "topic": content_item.get("topic", "Information"),
                                    "summary": content_item.get(
                                        "summary", str(content_item)
                                    ),
                                }
                            )
                        elif isinstance(content_item, str):
                            content_list.append(
                                {"topic": "Content", "summary": content_item}
                            )
                        elif isinstance(content_item, list):
                            for sub_item in content_item:
                                if isinstance(sub_item, dict):
                                    content_list.append(
                                        {
                                            "topic": sub_item.get(
                                                "topic", "Information"
                                            ),
                                            "summary": sub_item.get(
                                                "summary", str(sub_item)
                                            ),
                                        }
                                    )
                                else:
                                    content_list.append(
                                        {"topic": "Content", "summary": str(sub_item)}
                                    )
                except Exception as e:
                    logger.error(f"Error parsing extracted_content: {e}")
                    content_list = []

                results.append(
                    {
                        "url": url,
                        "title": item.get("metadata", {}).get("title", ""),
                        "content": content_list,
                        "images": image_list,
                        "videos": video_list,
                    }
                )
                all_images.extend(image_list)
                all_videos.extend(video_list)

                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "citation",
                            "data": {
                                "document": [f"Content from {url}"],
                                "metadata": [{"source": url}],
                                "source": {
                                    "name": item.get("metadata", {}).get("title", url)
                                },
                            },
                        }
                    )

            self.content_counter += len(results)
            if __event_emitter__ and self.valves.MORE_STATUS:
                s = "s" if self.content_counter > 1 else ""
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"Analyzed {self.content_counter} page{s} from {self.total_urls} URLs...",
                            "done": False,
                        },
                    }
                )

            if self.valves.DEBUG:
                logger.info(f"Successfully crawled {len(results)} URLs")

            # Normalize content in each result before returning
            normalized_results = []
            for result in results:
                r = result.copy()
                if "content" in r:
                    r["content"] = self._normalize_content(r["content"])
                normalized_results.append(r)

            return {
                "content": normalized_results,
                "images": all_images or [],
                "videos": all_videos or [],
                "links": all_links if extract_links else [],
            }

        except requests.exceptions.ConnectionError as e:
            error_msg = (
                f"Cannot connect to Crawl4AI at {endpoint}. Please verify:\n"
                f"- Crawl4AI container is running\n"
                f"- CRAWL4AI_BASE_URL is correct: '{self.valves.CRAWL4AI_BASE_URL}'\n"
                f"- Network connectivity between containers\n"
                f"Original error: {str(e)}"
            )
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "error", "data": {"description": error_msg, "done": True}}
                )
            return {
                "error": error_msg,
                "details": str(e),
                "content": [],
                "images": [],
                "videos": [],
            }

        except requests.exceptions.Timeout as e:
            timeout_seconds = self.valves.CRAWL4AI_TIMEOUT * len(urls) + 60
            error_msg = (
                f"Timeout connecting to Crawl4AI at {endpoint} after {timeout_seconds}s. "
                f"Check if Crawl4AI is processing too many URLs."
            )
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "error", "data": {"description": error_msg, "done": True}}
                )
            return {
                "error": error_msg,
                "details": str(e),
                "content": [],
                "images": [],
                "videos": [],
            }

        except requests.exceptions.RequestException as e:
            error_msg = (
                f"Network error connecting to Crawl4AI: {str(e)}. "
                f"Check if '{self.valves.CRAWL4AI_BASE_URL}' is accessible."
            )
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "error", "data": {"description": error_msg, "done": True}}
                )
            return {
                "error": error_msg,
                "details": str(e),
                "content": [],
                "images": [],
                "videos": [],
            }

        except Exception as e:
            error_msg = (
                f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
            )
            logger.error(error_msg)
            if __event_emitter__:
                await __event_emitter__(
                    {"type": "error", "data": {"description": error_msg, "done": True}}
                )
            return {
                "error": error_msg,
                "details": str(e),
                "content": [],
                "images": [],
                "videos": [],
            }

    # endregion
