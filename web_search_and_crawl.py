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

# ==================== REGION: IMPORTS ====================
import os
import socket
import traceback
import requests
import orjson
import tiktoken
import aiohttp
import asyncio
from urllib.parse import parse_qs, urlparse, quote
from pydantic import BaseModel, Field, model_validator
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



# ==================== REGION: AUXILIARY CLASSES ====================
class ArticleData(BaseModel):
    """Schema for LLM-extracted content."""
    topic: str
    summary: str


class ResearchCrawlMode:
    """Enumeration of research crawling modes."""

    PSEUDO_ADAPTIVE = "pseudo_adaptive"
    LLM_GUIDED = "llm_guided"
    BFS_DEEP = "bfs_deep"
    RESEARCH_FILTER = "research_filter"



# ==================== REGION: MAIN CLASS Tools ====================
class Tools:

    # -------------------- Subclass Valves (global configuration) --------------------
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
            description="The full URL for your SearXNG API instance. Insert <query> where the search terms should go. Include http:// or https:// prefix.",
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
            description="The base URL for your Crawl4AI instance. Include http:// or https:// prefix.",
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
            description="The number of URLs to send to Crawl4AI per batch. If more than this number of URLs are found in total, the tool will send them to Crawl4AI in batches of this number.",
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
            description="Only extract text content, excluding images and other media. Disables crawling and displaying media in the chat.",
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
            description="Image thumbnail size (in px) square. Ignored if 'Display images as thumbnails' is off.",
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
            title="Max Tokens used by web content",
            default=0,
            description="Maximum tokens to use for the web search content response. Set to 0 for unlimited.",
        )
        LLM_BASE_URL: str = Field(
            title="LLM Base URL",
            default="https://openrouter.ai/api/v1",
            description="The base URL for your preferred OpenAI-compatible LLM. Include http:// or https:// prefix.",
        )
        LLM_API_TOKEN: str = Field(
            title="LLM API Token",
            default="",
            description="Optional API Token for your preferred OpenAI-compatible LLM.",
        )
        LLM_PROVIDER: str = Field(
            title="LLM Provider and model",
            default="openrouter/@preset/default",
            description="The LLM provider and model to use. Format: <provider>/<model-name>. Examples: openai/gpt-4o, ollama/llama3.2, openrouter/@preset/default.",
        )
        LLM_TEMPERATURE: float = Field(
            title="LLM Temperature",
            default=0.3,
            description="The temperature to use for the LLM.",
        )
        LLM_INSTRUCTION: str = Field(
            title="LLM Extraction Instruction",
            default="Focus on extracting the core content. Summarize lengthy sections into concise points. Include: key concepts, examples, critical details, data from tables. Exclude: navigation, sidebars, footers, ads, comments, non-essential information. Format as clean markdown with code blocks and headers.",
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

        @model_validator(mode='after')
        def validate_settings(self):
            """Validate the conditional settings."""
            if not self.USE_NATIVE_SEARCH and not self.SEARCH_WITH_SEARXNG:
                raise ValueError(
                    "Either 'Use Native Search' or 'Search with SearXNG' must be enabled"
                )
            if self.SEARCH_WITH_SEARXNG and (not self.SEARXNG_BASE_URL or not self.SEARXNG_BASE_URL.strip()):
                raise ValueError(
                    "'SearXNG Search URL' is required when 'Search with SearXNG' is enabled. "
                    "Please provide the URL for your SearXNG instance."
                )
            return self



    # -------------------- Subclass UserValves (per-user configuration) --------------------
    class UserValves(BaseModel):
        SEARXNG_MAX_RESULTS: int = Field(
            title="SearXNG Max Results",
            default=None,
            description="The maximum number of results to return from SearXNG.",
        )
        CRAWL4AI_MAX_URLS: int = Field(
            title="Crawl4AI Maximum URLs to crawl",
            default=None,
            description="The maximum number of URLs to crawl with Crawl4AI.",
        )
        CRAWL4AI_DISPLAY_MEDIA: bool = Field(
            title="Display Media in Chat",
            default=None,
            description="Display images and videos as clickable links in the chat window.",
        )
        CRAWL4AI_MAX_MEDIA_ITEMS: int = Field(
            title="Max Media Items to Display",
            default=None,
            description="Maximum number of images/videos to display (0 = unlimited).",
        )
        CRAWL4AI_DISPLAY_THUMBNAILS: bool = Field(
            title="Display images as thumbnails",
            default=None,
            description="Display images as thumbnails in the chat window. Turn off to display images full-sized.",
        )
        CRAWL4AI_THUMBNAIL_SIZE: int = Field(
            title="Image thumbnail size",
            default=None,
            description="Image thumbnail size (in px) square. Ignored if 'Display images as thumbnails' is off.",
        )
        RESEARCH_MODE: bool = Field(
            default=False,
            description="Enable research mode using Crawl4AI with Deep Crawling.",
        )
        RESEARCH_CRAWL_MODE: Literal[
            "pseudo_adaptive", "llm_guided", "bfs_deep", "research_filter"
        ] = Field(
            default="pseudo_adaptive",
            description="The crawling strategy to use in Research Mode: pseudo_adaptive (keyword-based scoring), llm_guided (LLM selects links), bfs_deep (breadth-first), research_filter (URL filtering).",
        )
        RESEARCH_KEYWORD_WEIGHT: float = Field(
            default=0.7,
            description="The keyword relevance weight when using Research mode.",
        )
        RESEARCH_MAX_DEPTH: int = Field(
            default=2,
            le=10,
            description="The maximum depth of links to follow for the Research mode. CAUTION: Too high a value may cause excessive crawling.",
        )
        RESEARCH_MAX_PAGES: int = Field(
            default=15,
            le=25,
            description="The maximum number of pages to crawl in Research mode. CAUTION: Too high a value may cause excessive crawling.",
        )
        RESEARCH_BATCH_SIZE: int = Field(
            default=5,
            description="Number of URLs to process per batch during research crawling.",
        )
        RESEARCH_LLM_LINK_SELECTION: bool = Field(
            default=True,
            description="Use LLM to select next links when in llm_guided mode.",
        )
        RESEARCH_INCLUDE_EXTERNAL: bool = Field(
            default=False,
            description="Allow following external domains during research crawling.",
        )



    # -------------------- Initialization and Auto-configuration --------------------
    def __init__(self):
        self.valves = self.Valves()
        self.user_valves = self.UserValves()

        self._auto_configure()

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
            self.valves.SEARXNG_BASE_URL = f"{searxng_parsed_url.scheme}://{searxng_parsed_url.netloc}{searxng_parsed_url.path}?{reconstructed_query}"

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


    def _normalize_url(self, url: str, default_protocol: str = "http://") -> str:
        """Normalize URLs ensuring they have protocol and proper format."""
        if not url or not isinstance(url, str):
            return url

        url = url.strip()

        if not url.startswith(("http://", "https://")):
            logger.warning(
                f"URL '{url}' missing protocol. Adding '{default_protocol}'. "
                "Consider updating your configuration to include http:// or https:// prefix."
            )
            url = f"{default_protocol}{url}"

        url = url.rstrip("/")
        return url


    def _validate_llm_provider(self):
        """Validate and correct LLM provider format."""
        provider = self.valves.LLM_PROVIDER
        if not provider:
            logger.warning("LLM_PROVIDER is not set")
            return

        valid_providers = [
            "ollama/",
            "openai/",
            "openrouter/",
            "anthropic/",
            "azure/",
            "groq/",
            "cohere/",
        ]

        if any(provider.startswith(p) for p in valid_providers):
            return

        if (
            "11434" in self.valves.LLM_BASE_URL
            or "ollama" in self.valves.LLM_BASE_URL.lower()
        ):
            corrected = f"ollama/{provider}"
            logger.warning(
                f"LLM_PROVIDER '{provider}' looks like Ollama. Auto-correcting to '{corrected}'"
            )
            self.valves.LLM_PROVIDER = corrected
        else:
            logger.warning(
                f"LLM_PROVIDER '{provider}' may be missing provider prefix. Expected format: provider/model"
            )


    def _auto_configure(self):
        """Auto-detects configuration based on environment."""
        in_docker = os.path.exists("/.dockerenv")
        if in_docker and self.valves.DEBUG:
            logger.info("Running in Docker environment")

        self.valves.CRAWL4AI_BASE_URL = self._normalize_url(self.valves.CRAWL4AI_BASE_URL)
        self.valves.SEARXNG_BASE_URL = self._normalize_url(self.valves.SEARXNG_BASE_URL)
        self.valves.LLM_BASE_URL = self._normalize_url(self.valves.LLM_BASE_URL)

        original_base_url = self.valves.LLM_BASE_URL
        if (
            self.valves.LLM_BASE_URL == "http://openrouter.ai/api/v1"
            or self.valves.LLM_BASE_URL == "https://openrouter.ai/api/v1"
        ):
            try:
                socket.gethostbyname("host.docker.internal")
                self.valves.LLM_BASE_URL = "http://host.docker.internal:11434"
                logger.info(
                    "Auto-configured LLM_BASE_URL for Ollama via host.docker.internal"
                )
            except:
                try:
                    socket.gethostbyname("localhost")
                    self.valves.LLM_BASE_URL = "http://localhost:11434"
                    logger.info("Auto-configured LLM_BASE_URL for Ollama via localhost")
                except:
                    logger.info(
                        "Could not auto-detect Ollama, keeping original LLM_BASE_URL"
                    )

        if self.valves.LLM_BASE_URL != original_base_url:
            logger.warning(
                f"LLM_BASE_URL was automatically changed from '{original_base_url}' to '{self.valves.LLM_BASE_URL}'. "
                "If this is not desired, set a custom URL in the valve configuration."
            )

        self._validate_llm_provider()

        logger.info("Web Search and Crawl tool initialized with:")
        logger.info(f"  - Crawl4AI URL: {self.valves.CRAWL4AI_BASE_URL}")
        logger.info(f"  - LLM Provider: {self.valves.LLM_PROVIDER}")
        logger.info(f"  - LLM Base URL: {self.valves.LLM_BASE_URL}")
        logger.info(f"  - Native Search: {self.valves.USE_NATIVE_SEARCH}")
        logger.info(f"  - SearXNG: {self.valves.SEARCH_WITH_SEARXNG}")



    # -------------------- Helpers for Notifications and Logging --------------------
    async def _emit_status(self, description: str, done: bool = False, emitter: Callable = None):
        """Emit a status notification if emitter is provided."""
        if emitter:
            await emitter({"type": "status", "data": {"description": description, "done": done}})


    async def _emit_message_delta(self, content: str, emitter: Callable = None):
        """Emit a chat message delta if emitter is provided."""
        if emitter and content:
            await emitter({"type": "chat:message:delta", "data": {"content": content}})


    async def _emit_message(self, content: str, emitter: Callable = None):
        """Emit a full message if emitter is provided."""
        if emitter and content:
            await emitter({"type": "message", "data": {"content": content}})


    async def _log_and_emit_status(self, message: str, level: str = "info", done: bool = False, emitter: Callable = None):
        """Log a message and emit a status notification."""
        log_method = getattr(logger, level, logger.info)
        log_method(message)
        await self._emit_status(message, done=done, emitter=emitter)


    async def _log_and_emit_message(self, message: str, level: str = "info", emitter: Callable = None):
        """Log a message and emit a full message."""
        log_method = getattr(logger, level, logger.info)
        log_method(message)
        await self._emit_message(message, emitter=emitter)


    async def _log_and_emit_message_delta(self, message: str, level: str = "info", emitter: Callable = None):
        """Log a message and emit a message delta."""
        if message:
            log_method = getattr(logger, level, logger.info)
            log_method(message)
            await self._emit_message_delta(message, emitter=emitter)



    # -------------------- URL and Content Normalization --------------------
    def _normalize_content(self, content_items: List[Any]) -> List[dict]:
        """Normalize content to consistent dictionary format with topic and summary."""
        normalized = []
        for item in content_items:
            if isinstance(item, dict):
                topic = item.get("topic", item.get("title", "Content"))
                summary = item.get("summary", item.get("content", ""))

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
                normalized.extend(self._normalize_content(item))
            else:
                normalized.append({"topic": "Content", "summary": str(item)})
        return normalized



    # -------------------- Token Counting and Truncation --------------------
    async def _count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """Count tokens in text using tiktoken."""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


    async def _truncate_content(self, content: str, max_tokens: int, model: str = "gpt-4") -> str:
        """Truncate content to fit within max_tokens."""
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



    # -------------------- Image Validation --------------------
    async def _validate_image_url(self, url: str) -> bool:
        """Validate if an image URL is accessible and returns an image."""
        try:
            if not self.valves.CRAWL4AI_VALIDATE_IMAGES:
                return True

            timeout = aiohttp.ClientTimeout(total=4)
            url = url.strip()
            headers = {
                "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
            async with aiohttp.ClientSession(
                timeout=timeout,
                headers=headers,
                skip_auto_headers={"Accept-Encoding", "Content-Type"},
            ) as session:
                async with session.head(url, allow_redirects=True) as response:
                    if response.status != 200:
                        logger.warning(f"Image validation failed for {url}: Status {response.status}")
                        return False
                    content_type = response.headers.get("Content-Type", "").lower()
                    if not content_type.startswith("image/"):
                        logger.warning(f"Image validation failed for {url}: Content-Type {content_type}")
                        return False
                    return True
        except asyncio.TimeoutError:
            logger.warning(f"Image validation timeout for {url}")
            return False
        except Exception as e:
            logger.warning(f"Image validation error for {url}: {str(e)}")
            return False


    async def _validate_images_batch(self, urls: List[str]) -> List[str]:
        """Validate multiple image URLs concurrently. Returns list of valid URLs only."""
        tasks = [self._validate_image_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        valid_urls = [url for url, is_valid in zip(urls, results) if is_valid]
        if len(valid_urls) < len(urls) and self.valves.DEBUG:
            logger.info(f"Image validation: {len(valid_urls)}/{len(urls)} images are valid")
        return valid_urls



    # -------------------- Native Search (OpenWebUI) --------------------
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

            if self.valves.MORE_STATUS:
                await self._log_and_emit_status("Searching using Open WebUI native search...", done=False, emitter=__event_emitter__)

            form = SearchForm.model_validate({"queries": [query]})
            result = await process_web_search(
                request=Request(scope={"type": "http", "app": app}),
                form_data=form,
                user=user,
            )
            if self.valves.DEBUG:
                logger.info(f"Native search for '{query}' returned {result}")

            urls = [item.get("link") for item in result.get("items", []) if item.get("link")]

            if self.valves.DEBUG:
                logger.info(f"Native search for '{query}' returned {len(urls)} URLs")

            if self.valves.MORE_STATUS:
                await self._log_and_emit_status(f"Found {len(urls)} websites...", done=False, emitter=__event_emitter__)

            return urls

        except Exception as e:
            error_msg = f"Native search encountered an error: {str(e)}"
            await self._log_and_emit_status(error_msg, level="error", done=False, emitter=__event_emitter__)
            return []



    # -------------------- SearXNG Search --------------------
    async def _search_searxng(
        self, query: str, __event_emitter__: Callable[[dict], Any] = None
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

        if self.valves.MORE_STATUS:
            await self._log_and_emit_status("Searching using SearXNG...", done=False, emitter=__event_emitter__)

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
            urls = []
            max_results = (
                self.user_valves.SEARXNG_MAX_RESULTS or self.valves.SEARXNG_MAX_RESULTS
            )
            for result in results[:max_results]:
                if result.get("url"):
                    urls.append(result["url"])

            if self.valves.DEBUG:
                logger.info(f"SearXNG search for '{query}' returned {len(urls)} URLs")

            if self.valves.MORE_STATUS:
                await self._log_and_emit_status(f"Found {len(urls)} results...", done=False, emitter=__event_emitter__)

            return urls

        except requests.exceptions.RequestException as e:
            error_msg = f"SearXNG search error: {str(e)}"
            await self._log_and_emit_status(error_msg, level="error", done=False, emitter=__event_emitter__)
            return []
        except Exception as e:
            logger.error(f"Unexpected error in SearXNG search: {str(e)}")
            return []



    # -------------------- Main Entry Point: search_and_crawl --------------------
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
        Main method called by the LLM. Performs search, crawls URLs, and returns content.
        """
        logger.info(f"Starting search and crawl for '{query}'")

        # Validate configuration and notify user of issues
        await self._notify_configuration_issues(__event_emitter__)

        # Critical validation
        if not self.valves.USE_NATIVE_SEARCH and not self.valves.SEARCH_WITH_SEARXNG:
            error_msg = "❌ Configuration error: Neither Native Search nor SearXNG is enabled. Please enable at least one search source in the tool settings."
            await self._log_and_emit_status(error_msg, level="error", done=True, emitter=__event_emitter__)
            return error_msg

        if self.valves.SEARCH_WITH_SEARXNG and (not self.valves.SEARXNG_BASE_URL or not self.valves.SEARXNG_BASE_URL.strip()):
            error_msg = "❌ Configuration error: SearXNG is enabled but the base URL is empty. Please set SEARXNG_BASE_URL in the tool settings."
            await self._log_and_emit_status(error_msg, level="error", done=True, emitter=__event_emitter__)
            return error_msg

        gathered_urls = []
        self.crawl_counter = 0
        self.content_counter = 0
        self.total_urls = 0

        if not max_images:
            max_images = (
                self.user_valves.CRAWL4AI_MAX_MEDIA_ITEMS
                or self.valves.CRAWL4AI_MAX_MEDIA_ITEMS
            )

        # Add user-provided URLs
        if urls:
            for url in urls:
                if not url.startswith("http"):
                    url = f"https://{url}"
                if url not in gathered_urls:
                    gathered_urls.append(url)

        await self._log_and_emit_message_delta(self.valves.INITIAL_RESPONSE.strip(), emitter=__event_emitter__)
        await self._log_and_emit_status(f"Searching for '{query}'...", done=False, emitter=__event_emitter__)

        # Search with Native Search
        if self.valves.USE_NATIVE_SEARCH:
            native_urls = await self._search_native(query, __event_emitter__, __user__)
            for url in native_urls:
                if url not in gathered_urls:
                    gathered_urls.append(url)

        # Search with SearXNG
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
            msg = f"Nothing found for query '{query}'."
            await self._log_and_emit_status(msg, done=True, emitter=__event_emitter__)
            return f"No URLs found to crawl for the query: {query}."

        max_urls = self.user_valves.CRAWL4AI_MAX_URLS or self.valves.CRAWL4AI_MAX_URLS
        if len(gathered_urls) > max_urls:
            gathered_urls = gathered_urls[:max_urls]

        if self.valves.MORE_STATUS:
            await self._log_and_emit_status(f"Found {len(gathered_urls)} results. Inspecting the content...", done=False, emitter=__event_emitter__)

        effective_research_mode = research_mode or self.user_valves.RESEARCH_MODE
        effective_crawl_mode = research_crawl_mode or self.user_valves.RESEARCH_CRAWL_MODE

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

        # Research mode or standard batch crawling
        if effective_research_mode and len(gathered_urls) > 0:
            if self.valves.MORE_STATUS:
                await self._log_and_emit_status(f"Research Mode enabled. Using '{effective_crawl_mode}' strategy...", done=False, emitter=__event_emitter__)

            research_result = await self._research_crawl(
                urls=gathered_urls,
                query=query,
                mode=effective_crawl_mode,
                __event_emitter__=__event_emitter__,
            )

            if "content" in research_result:
                normalized_content = self._normalize_content(research_result["content"])
                crawl_results.extend(normalized_content)
                if self.valves.DEBUG:
                    logger.info(f"Research mode added {len(normalized_content)} content items")
            if "images" in research_result:
                image_list.extend(research_result["images"])
            if "videos" in research_result:
                video_list.extend(research_result["videos"])
        else:
            for i in range(0, len(gathered_urls), self.valves.CRAWL4AI_BATCH):
                batch = gathered_urls[i : i + self.valves.CRAWL4AI_BATCH]
                try:
                    crawled_batch = await self._crawl_url(
                        urls=batch, query=query, __event_emitter__=__event_emitter__
                    )

                    if self.valves.DEBUG:
                        logger.info(
                            f"Found {len(crawled_batch.get('content',[]))} content, {len(crawled_batch.get('images',[]))} images, {len(crawled_batch.get('videos',[]))} videos."
                        )

                    # Images
                    if crawled_batch.get("images", []):
                        for img_url in crawled_batch["images"]:
                            parsed_image = urlparse(img_url)
                            base_image_url = f"{parsed_image.scheme}://{parsed_image.netloc}{parsed_image.path}"
                            if base_image_url in seen_images:
                                continue
                            seen_images.add(base_image_url)
                            thumbnail_url = f"https://images.weserv.nl/?url={quote(img_url)}&w={thumbnail_size}&h={thumbnail_size}&fit=inside"
                            image_valid = await self._validate_image_url(img_url)
                            thumbnail_valid = await self._validate_image_url(thumbnail_url)
                            if image_valid and thumbnail_valid:
                                image_list.append(img_url)

                    # Videos
                    if crawled_batch.get("videos", []):
                        for vid_url in crawled_batch["videos"]:
                            parsed_video = urlparse(vid_url)
                            base_video_url = f"{parsed_video.scheme}://{parsed_video.netloc}{parsed_video.path}"
                            if base_video_url in seen_videos:
                                continue
                            seen_videos.add(base_video_url)
                            video_list.append(vid_url)

                    # Content
                    data_list = crawled_batch.get("content", [])
                    normalized_data_list = self._normalize_content(data_list)

                    if normalized_data_list:
                        content_str = orjson.dumps(normalized_data_list).decode("utf-8")
                        page_tokens = await self._count_tokens(content_str)

                        if self.valves.CRAWL4AI_MAX_TOKENS > 0 and page_tokens > self.valves.CRAWL4AI_MAX_TOKENS:
                            content_str = await self._truncate_content(content_str, self.valves.CRAWL4AI_MAX_TOKENS)
                            try:
                                normalized_data_list = orjson.loads(
                                    content_str.replace("\n\n[Content truncated due to length...]", "")
                                )
                            except:
                                pass
                            page_tokens = self.valves.CRAWL4AI_MAX_TOKENS
                            if self.valves.DEBUG:
                                logger.info(f"Truncated content from batch to {self.valves.CRAWL4AI_MAX_TOKENS} tokens")

                            if total_tokens + page_tokens > self.valves.CRAWL4AI_MAX_TOKENS:
                                logger.warning(f"Reached token limit ({self.valves.CRAWL4AI_MAX_TOKENS}). Skipping remaining pages.")
                                if self.valves.MORE_STATUS:
                                    await self._log_and_emit_status(
                                        f"Token limit reached. Processed {len(crawl_results)} of {len(data_list)} pages.",
                                        level="warning",
                                        done=False,
                                        emitter=__event_emitter__
                                    )
                                continue

                        total_tokens += page_tokens
                        if self.valves.DEBUG:
                            logger.info(
                                f"Batch {batch_count}: {page_tokens} tokens (Total: {total_tokens}/{self.valves.CRAWL4AI_MAX_TOKENS if self.valves.CRAWL4AI_MAX_TOKENS > 0 else 'unlimited'})"
                            )
                        crawl_results.extend(normalized_data_list)

                    batch_count += 1

                except Exception as e:
                    error_message = f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
                    logger.error(error_message)

        # Final normalization
        crawl_results = self._normalize_content(crawl_results)

        if self.valves.DEBUG:
            logger.info(f"Final crawl_results count: {len(crawl_results)}")
            for idx, item in enumerate(crawl_results[:3]):
                logger.info(f"Sample {idx}: {type(item)} - {str(item)[:100]}")

        # Display media if enabled
        if self.user_valves.CRAWL4AI_DISPLAY_MEDIA or self.valves.CRAWL4AI_DISPLAY_MEDIA:
            max_items = self.valves.CRAWL4AI_MAX_MEDIA_ITEMS
            image_list = image_list[:max_images] if max_images > 0 else image_list
            video_list = video_list[:max_items] if max_items > 0 else video_list

            if image_list:
                image_markdown = ""
                for img_url in image_list:
                    if (self.user_valves.CRAWL4AI_DISPLAY_THUMBNAILS or self.valves.CRAWL4AI_DISPLAY_THUMBNAILS):
                        thumbnail_url = f"https://images.weserv.nl/?url={quote(img_url)}&w={thumbnail_size}&h={thumbnail_size}&fit=inside"
                    else:
                        thumbnail_url = img_url
                    image_markdown += f"[![image]({thumbnail_url})]({img_url})\n"
                await self._emit_message(image_markdown, emitter=__event_emitter__)

            if video_list:
                video_markdown = "\n\n*Videos links:*\n"
                for idx, vid_url in enumerate(video_list, 1):
                    video_markdown += f"{idx}. [{vid_url}]({vid_url})\n"
                await self._emit_message(video_markdown, emitter=__event_emitter__)

        await self._log_and_emit_status(f"Inspected {len(crawl_results)} web pages.", done=True, emitter=__event_emitter__)

        return crawl_results



    # -------------------- Research Mode Routing --------------------
    async def _research_crawl(
        self,
        urls: List[str],
        query: str,
        mode: str = "pseudo_adaptive",
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """Route to the appropriate research crawling strategy."""
        if mode == ResearchCrawlMode.PSEUDO_ADAPTIVE:
            return await self._pseudo_adaptive_crawl(urls, query, __event_emitter__)
        elif mode == ResearchCrawlMode.LLM_GUIDED:
            return await self._llm_guided_crawl(urls, query, __event_emitter__)
        elif mode == ResearchCrawlMode.BFS_DEEP:
            return await self._bfs_deep_crawl(urls, query, __event_emitter__)
        elif mode == ResearchCrawlMode.RESEARCH_FILTER:
            return await self._research_filter_crawl(urls, query, __event_emitter__)
        else:
            logger.warning(f"Unknown research crawl mode: {mode}, defaulting to pseudo_adaptive")
            return await self._pseudo_adaptive_crawl(urls, query, __event_emitter__)



    # -------------------- Research Mode Strategies --------------------
    async def _pseudo_adaptive_crawl(
        self,
        start_urls: List[str],
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """Keyword-based adaptive crawling."""
        from collections import deque
        from urllib.parse import urlparse

        max_pages = self.user_valves.RESEARCH_MAX_PAGES
        max_depth = self.user_valves.RESEARCH_MAX_DEPTH
        batch_size = self.user_valves.RESEARCH_BATCH_SIZE
        include_external = self.user_valves.RESEARCH_INCLUDE_EXTERNAL

        keywords = query.lower().split()
        crawled_pages = set()
        crawled_results = []
        all_images = []
        all_videos = []
        queue = deque()

        for url in start_urls[:5]:
            if url not in crawled_pages:
                score = sum(1 for kw in keywords if kw in url.lower())
                queue.append((url, 0, score))

        self.total_urls = max_pages

        while queue and len(crawled_pages) < max_pages:
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

                if self.valves.MORE_STATUS:
                    await self._log_and_emit_status(
                        f"[Pseudo-Adaptive] Depth {depth}: Crawling {url[:60]}... ({len(crawled_pages)}/{max_pages})",
                        done=False,
                        emitter=__event_emitter__
                    )

                result = await self._crawl_url(
                    urls=[url],
                    query=query,
                    extract_links=True,
                    __event_emitter__=__event_emitter__,
                )

                if result.get("content"):
                    crawled_results.extend(self._normalize_content(result["content"]))
                if result.get("images"):
                    all_images.extend(result["images"])
                if result.get("videos"):
                    all_videos.extend(result["videos"])

                if depth < max_depth:
                    discovered_links = result.get("links", [])
                    for link in discovered_links:
                        if link in crawled_pages:
                            continue
                        parsed_link = urlparse(link)
                        parsed_url = urlparse(url)
                        if not include_external:
                            if parsed_link.netloc and parsed_link.netloc != parsed_url.netloc:
                                continue
                        link_lower = link.lower()
                        link_score = sum(1 for kw in keywords if kw in link_lower)
                        if link_score > 0:
                            queue.append((link, depth + 1, link_score))

        if self.valves.DEBUG:
            logger.info(f"[Pseudo-Adaptive] Crawled {len(crawled_pages)} pages")

        return {
            "content": crawled_results,
            "images": all_images,
            "videos": all_videos,
            "pages_crawled": len(crawled_pages),
        }


    async def _llm_guided_crawl(
        self,
        start_urls: List[str],
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """LLM-guided link selection (currently uses keyword fallback)."""
        from urllib.parse import urlparse

        max_pages = self.user_valves.RESEARCH_MAX_PAGES
        use_llm_selection = self.user_valves.RESEARCH_LLM_LINK_SELECTION
        include_external = self.user_valves.RESEARCH_INCLUDE_EXTERNAL

        crawled_pages = set()
        crawled_results = []
        all_images = []
        all_videos = []

        llm_config = LLMConfig(
            provider=self.valves.LLM_PROVIDER,
            base_url=self.valves.LLM_BASE_URL,
            temperature=0.3,
            max_tokens=500,
        )
        if self.valves.LLM_API_TOKEN:
            llm_config.api_token = self.valves.LLM_API_TOKEN

        urls_to_process = list(start_urls[:5])

        while urls_to_process and len(crawled_pages) < max_pages:
            current_url = urls_to_process.pop(0)
            if current_url in crawled_pages:
                continue

            crawled_pages.add(current_url)

            if self.valves.MORE_STATUS:
                await self._log_and_emit_status(
                    f"[LLM-Guided] Crawling {current_url[:60]}... ({len(crawled_pages)}/{max_pages})",
                    done=False,
                    emitter=__event_emitter__
                )

            result = await self._crawl_url(
                urls=[current_url],
                query=query,
                extract_links=True,
                __event_emitter__=__event_emitter__,
            )

            if result.get("content"):
                crawled_results.extend(self._normalize_content(result["content"]))
            if result.get("images"):
                all_images.extend(result["images"])
            if result.get("videos"):
                all_videos.extend(result["videos"])

            discovered_links = result.get("links", [])[:15]
            if not discovered_links:
                continue

            if not include_external:
                parsed_current = urlparse(current_url)
                filtered_links = []
                for link in discovered_links:
                    parsed_link = urlparse(link)
                    if not parsed_link.netloc or parsed_link.netloc == parsed_current.netloc:
                        filtered_links.append(link)
                discovered_links = filtered_links

            if not discovered_links:
                continue

            if use_llm_selection:
                # LLM selection placeholder (currently falls back to keyword scoring)
                pass

            keywords = query.lower().split()
            scored_links = []
            for link in discovered_links:
                if link in crawled_pages or link in urls_to_process:
                    continue
                link_lower = link.lower()
                score = sum(1 for kw in keywords if kw in link_lower)
                if score > 0:
                    scored_links.append((link, score))

            scored_links.sort(key=lambda x: x[1], reverse=True)

            for link, score in scored_links[:3]:
                if link not in urls_to_process and link not in crawled_pages:
                    urls_to_process.append(link)

        if self.valves.DEBUG:
            logger.info(f"[LLM-Guided] Crawled {len(crawled_pages)} pages")

        return {
            "content": crawled_results,
            "images": all_images,
            "videos": all_videos,
            "pages_crawled": len(crawled_pages),
        }


    async def _bfs_deep_crawl(
        self,
        start_urls: List[str],
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """Breadth-first deep crawling."""
        from collections import deque
        from urllib.parse import urlparse

        max_pages = self.user_valves.RESEARCH_MAX_PAGES
        max_depth = self.user_valves.RESEARCH_MAX_DEPTH
        batch_size = self.user_valves.RESEARCH_BATCH_SIZE
        include_external = self.user_valves.RESEARCH_INCLUDE_EXTERNAL

        crawled_pages = set()
        crawled_results = []
        all_images = []
        all_videos = []

        if start_urls:
            parsed_start = urlparse(start_urls[0])
            base_domain = parsed_start.netloc
        else:
            base_domain = ""

        queue = deque()
        for url in start_urls[:5]:
            if url not in crawled_pages:
                queue.append((url, 0))

        self.total_urls = max_pages

        while queue and len(crawled_pages) < max_pages:
            level_size = min(batch_size, len(queue))
            level_batch = []
            for _ in range(level_size):
                if queue:
                    level_batch.append(queue.popleft())

            for url, depth in level_batch:
                if len(crawled_pages) >= max_pages:
                    break
                if url in crawled_pages:
                    continue
                if depth > max_depth:
                    continue

                crawled_pages.add(url)

                if self.valves.MORE_STATUS:
                    await self._log_and_emit_status(
                        f"[BFS-Deep] Depth {depth}: Crawling {url[:60]}... ({len(crawled_pages)}/{max_pages})",
                        done=False,
                        emitter=__event_emitter__
                    )

                result = await self._crawl_url(
                    urls=[url],
                    query=query,
                    extract_links=True,
                    __event_emitter__=__event_emitter__,
                )

                if result.get("content"):
                    crawled_results.extend(self._normalize_content(result["content"]))
                if result.get("images"):
                    all_images.extend(result["images"])
                if result.get("videos"):
                    all_videos.extend(result["videos"])

                if depth < max_depth:
                    discovered_links = result.get("links", [])
                    for link in discovered_links[:10]:
                        if link in crawled_pages:
                            continue
                        parsed_link = urlparse(link)
                        if not include_external:
                            if parsed_link.netloc and parsed_link.netloc != base_domain:
                                continue
                        if link not in queue:
                            queue.append((link, depth + 1))

        if self.valves.DEBUG:
            logger.info(f"[BFS-Deep] Crawled {len(crawled_pages)} pages")

        return {
            "content": crawled_results,
            "images": all_images,
            "videos": all_videos,
            "pages_crawled": len(crawled_pages),
        }


    async def _research_filter_crawl(
        self,
        start_urls: List[str],
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """Research with relevance filtering."""
        max_pages = self.user_valves.RESEARCH_MAX_PAGES
        include_external = self.user_valves.RESEARCH_INCLUDE_EXTERNAL
        keywords = query.lower().split()

        results = {
            "content": [],
            "images": [],
            "videos": [],
            "sources": {},
            "total_pages": 0
        }

        for source_url in start_urls[:5]:
            if results["total_pages"] >= max_pages:
                break

            if self.valves.MORE_STATUS:
                await self._log_and_emit_status(
                    f"[Research-Filter] Researching: {source_url[:60]}... ({results['total_pages']}/{max_pages})",
                    done=False,
                    emitter=__event_emitter__
                )

            source_result = await self._crawl_url(
                urls=[source_url], query=query, extract_links=True, __event_emitter__=__event_emitter__
            )

            if source_result.get("content"):
                content_text = str(source_result["content"])
                relevance_score = sum(1 for kw in keywords if kw in content_text.lower())
                results["sources"][source_url] = {
                    "content": source_result["content"],
                    "relevance_score": relevance_score,
                    "links": source_result.get("links", [])[:10]
                }
                results["content"].extend(self._normalize_content(source_result["content"]))
                results["total_pages"] += 1

            if source_result.get("images"):
                results["images"].extend(source_result["images"])
            if source_result.get("videos"):
                results["videos"].extend(source_result["videos"])

            relevant_links = []
            for link in source_result.get("links", [])[:15]:
                if results["total_pages"] >= max_pages:
                    break
                link_lower = link.lower()
                score = sum(1 for kw in keywords if kw in link_lower)
                if score > 0:
                    relevant_links.append((link, score))

            relevant_links.sort(key=lambda x: x[1], reverse=True)

            crawled = 0
            max_links_per_source = 3
            for link, score in relevant_links:
                if results["total_pages"] >= max_pages:
                    break
                if crawled >= max_links_per_source:
                    break

                if not include_external:
                    from urllib.parse import urlparse
                    parsed_link = urlparse(link)
                    parsed_source = urlparse(source_url)
                    if parsed_link.netloc and parsed_link.netloc != parsed_source.netloc:
                        continue

                if self.valves.MORE_STATUS:
                    await self._log_and_emit_status(
                        f"[Research-Filter] Following: {link[:60]}...",
                        done=False,
                        emitter=__event_emitter__
                    )

                link_result = await self._crawl_url(
                    urls=[link], query=query, __event_emitter__=__event_emitter__
                )

                if link_result.get("content"):
                    results["content"].extend(self._normalize_content(link_result["content"]))
                    results["total_pages"] += 1
                    crawled += 1

                if link_result.get("images"):
                    results["images"].extend(link_result["images"])
                if link_result.get("videos"):
                    results["videos"].extend(link_result["videos"])

        results["content"].sort(
            key=lambda x: sum(1 for kw in keywords if kw in x.get("summary", "").lower()),
            reverse=True
        )

        if self.valves.DEBUG:
            logger.info(f"[Research-Filter] Crawled {results['total_pages']} pages")

        return results



    # -------------------- Core Crawling Function --------------------
    async def _crawl_url(
        self,
        urls: Union[list, str],
        query: Optional[str] = None,
        extract_links: bool = False,
        __event_emitter__: Callable[[dict], Any] = None
    ) -> dict:
        """Send URLs to Crawl4AI and return content, images, videos, and optionally links."""
        if isinstance(urls, str):
            urls = [urls]

        for idx, url in enumerate(urls):
            if not url.startswith("http"):
                urls[idx] = f"https://{url}"

        endpoint = f"{self.valves.CRAWL4AI_BASE_URL}/crawl"

        if self.valves.DEBUG:
            logger.info(f"Using LLM provider: {self.valves.LLM_PROVIDER}")

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
            base_url=self.valves.LLM_BASE_URL,
            temperature=self.valves.LLM_TEMPERATURE or 0.3,
            max_tokens=self.valves.LLM_MAX_TOKENS or None,
            top_p=self.valves.LLM_TOP_P or None,
            frequency_penalty=self.valves.LLM_FREQUENCY_PENALTY or None,
            presence_penalty=self.valves.LLM_PRESENCE_PENALTY or None
        )
        if self.valves.LLM_API_TOKEN:
            llm_config.api_token = self.valves.LLM_API_TOKEN

        extraction_strategy = LLMExtractionStrategy(
            llm_config=llm_config,
            instruction=self.valves.LLM_INSTRUCTION,
            input_format="fit_markdown",
            schema=ArticleData.model_json_schema(),
        )

        md_generator = DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(),
            options={"ignore_links": True, "escape_html": False, "body_width": 80}
        )

        crawler_config = CrawlerRunConfig(
            markdown_generator=md_generator,
            extraction_strategy=extraction_strategy,
            table_extraction=DefaultTableExtraction(),
            exclude_external_links=not self.valves.CRAWL4AI_EXTERNAL_DOMAINS,
            exclude_social_media_domains=[d.strip() for d in self.valves.CRAWL4AI_EXCLUDE_SOCIAL_MEDIA_DOMAINS.split(",") if d.strip()],
            exclude_domains=[d.strip() for d in self.valves.CRAWL4AI_EXCLUDE_DOMAINS.split(",") if d.strip()],
            user_agent=self.valves.CRAWL4AI_USER_AGENT,
            stream=False,
            cache_mode=CacheMode.BYPASS,
            page_timeout=self.valves.CRAWL4AI_TIMEOUT * 1000,
            only_text=self.valves.CRAWL4AI_TEXT_ONLY,
            word_count_threshold=self.valves.CRAWL4AI_WORD_COUNT_THRESHOLD,
            exclude_all_images=self.valves.CRAWL4AI_EXCLUDE_IMAGES == "All",
            exclude_external_images=self.valves.CRAWL4AI_EXCLUDE_IMAGES == "External",
        )

        if self.valves.MORE_STATUS and len(urls) > 1:
            await self._log_and_emit_status(f"Processing {len(urls)} URLs...", done=False, emitter=__event_emitter__)
        elif self.valves.MORE_STATUS:
            await self._log_and_emit_status(f"Processing {urls[0]}...", done=False, emitter=__event_emitter__)

        self.crawl_counter += len(urls)

        if self.valves.DEBUG:
            logger.info(f"Contacting Crawl4AI at {endpoint} for URLs: {urls}")

        headers = {"Content-Type": "application/json"}
        payload = {
            "urls": urls,
            "extraction_strategy": extraction_strategy.model_dump(mode="json"),
            "crawler_config": crawler_config.model_dump(mode="json"),
            "browser_config": browser_config.model_dump(mode="json"),
        }
        if extract_links:
            payload["extract_links"] = True

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.valves.CRAWL4AI_TIMEOUT)
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        logger.error(f"Crawl4AI error: {response.status} - {error_text}")
                        return {"content": [], "images": [], "videos": [], "links": [] if extract_links else None}

                    result = await response.json()
                    content = result.get("extracted_content", [])
                    images = result.get("images", [])
                    videos = result.get("videos", [])
                    links = result.get("links", []) if extract_links else None

                    if self.valves.CRAWL4AI_MIN_IMAGE_SCORE > 0 and images:
                        filtered_images = []
                        for img in images:
                            if isinstance(img, dict) and img.get("score", 0) >= self.valves.CRAWL4AI_MIN_IMAGE_SCORE:
                                filtered_images.append(img.get("src", ""))
                            elif isinstance(img, str):
                                filtered_images.append(img)
                        images = filtered_images

                    return {"content": content, "images": images, "videos": videos, "links": links}

        except asyncio.TimeoutError:
            logger.error(f"Timeout while crawling URLs: {urls}")
            return {"content": [], "images": [], "videos": [], "links": [] if extract_links else None}
        except Exception as e:
            logger.error(f"Unexpected error while crawling: {str(e)}\n{traceback.format_exc()}")
            return {"content": [], "images": [], "videos": [], "links": [] if extract_links else None}



    # -------------------- Configuration Notification Helper --------------------
    async def _notify_configuration_issues(self, __event_emitter__: Callable[[dict], Any] = None):
        """Check current valves for configuration issues and emit status notifications."""
        warnings = []

        for name, url in [
            ("Crawl4AI", self.valves.CRAWL4AI_BASE_URL),
            ("SearXNG", self.valves.SEARXNG_BASE_URL),
            ("LLM", self.valves.LLM_BASE_URL)
        ]:
            if url and not url.startswith(("http://", "https://")):
                warnings.append(f"⚠️ {name} URL missing protocol: '{url}'. It should start with http:// or https://. The tool may add it automatically, but please update your configuration.")

        provider = self.valves.LLM_PROVIDER
        if provider:
            valid_prefixes = ["ollama/", "openai/", "openrouter/", "anthropic/", "azure/", "groq/", "cohere/"]
            if not any(provider.startswith(p) for p in valid_prefixes):
                if "11434" in self.valves.LLM_BASE_URL or "ollama" in self.valves.LLM_BASE_URL.lower():
                    warnings.append(f"⚠️ LLM provider '{provider}' looks like Ollama. It should start with 'ollama/'. The tool may auto-correct it, but please update to 'ollama/{provider}'.")
                else:
                    warnings.append(f"⚠️ LLM provider '{provider}' may be missing a provider prefix. Expected format like 'openai/gpt-4o', 'ollama/llama3.2', etc. Please check your configuration.")

        if warnings and __event_emitter__:
            warning_msg = "\n".join(warnings)
            await self._log_and_emit_status(warning_msg, level="warning", done=False, emitter=__event_emitter__)
