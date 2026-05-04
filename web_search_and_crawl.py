"""
title: Web Search and Crawl
description: Search and Crawls the web using SearXNG, OpenWebUI Native Search, and Crawl4AI. Extracts content from URLs using a self-hosted Crawl4AI instance, optionally researching using Crawl4AI Deep Research.
author: lexiismadd, zeioth
author_url: https://github.com/lexiismad, https://github.com/zeioth
funding_url: https://github.com/open-webui
version: 2.8.10
license: MIT
requirements: aiohttp, loguru, crawl4ai, orjson, tiktoken
"""

# region ── Imports ────────────────────────────────────────────────────────────

import os
import re
import traceback
import anyio
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
        PREFLIGHT_TIMEOUT: int = Field(
            title="Pre-flight Check Timeout",
            default=8,
            description="Timeout in seconds for pre-flight HTML validation checks. Increase for slow sites, decrease for faster failure detection.",
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
        EXPANSION_LLM_PROVIDER: str = Field(
            title="LLM Provider for Query Expansion",
            default="",
            description="LLM provider/model to use for query expansion. If empty, falls back to LLM_PROVIDER.",
            examples=["ollama/llama3.2", "openai/gpt-4o-mini"],
        )
        FILTER_LLM_PROVIDER: str = Field(
            title="LLM Provider for URL Filtering",
            default="",
            description="LLM provider/model to use for URL filtering. If empty, falls back to LLM_PROVIDER.",
            examples=["ollama/llama3.2", "openai/gpt-4o-mini"],
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
            default=True,
            description="Show more status updates during web search and crawl",
        )
        DEBUG: bool = Field(
            title="Debug logging",
            default=True,
            description="Enable detailed debug logging",
        )
        USE_LLM_URL_FILTER: bool = Field(
            title="Use LLM URL Filter",
            default=False,
            description="Use LLM to analyze and filter out URLs that are obviously unrelated to the search query. This adds a small delay but improves relevance and saves tokens.",
        )
        LLM_URL_FILTER_CONCURRENCY: int = Field(
            title="LLM URL Filter Concurrency",
            default=5,
            description="Number of concurrent LLM requests for URL filtering.",
        )
        USE_QUERY_EXPANSION: bool = Field(
            title="Use Query Expansion",
            default=False,
            description="Use LLM to generate related search terms to find more relevant results.",
        )
        MAX_EXPANDED_QUERIES: int = Field(
            title="Max Expanded Queries",
            default=5,
            description="Maximum number of related search terms to generate for query expansion.",
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
                                "description": (
                                    "Enable deep research crawling. Use true for queries requiring multiple sources, "
                                    "comprehensive information gathering, or complex analysis.\n"
                                    "Use false for simple facts, quick lookups, recent news, or when a single page suffices.\n"
                                    "Leave unset to let the system decide based on your best judgment.\n\n"
                                    "When to use research mode:\n"
                                    "- Comparisons (e.g., 'compare X vs Y')\n"
                                    "- Historical overviews (e.g., 'history of the Roman Empire')\n"
                                    "- Latest developments in a field (e.g., 'best practices for microservices in 2025')\n"
                                    "- Multi‑faceted questions (e.g., 'impact of climate change on agriculture')\n\n"
                                    "When NOT to use research mode:\n"
                                    "- Simple definitions (e.g., 'define photosynthesis')\n"
                                    "- Current time, weather, or stock price\n"
                                    "- Single fact lookup (e.g., 'height of Mount Everest')\n"
                                    "- Recent breaking news (single event)"
                                ),
                                "default": None,
                            },
                            "research_crawl_mode": {
                                "type": "string",
                                "description": (
                                    "Crawling strategy if research mode is enabled:\n"
                                    "- pseudo_adaptive: for factual queries, news, or simple lookups.\n"
                                    "- llm_guided: when you need to select links based on actual content relevance.\n"
                                    "- bfs_deep: for exhaustive topic coverage (e.g., systematic reviews).\n"
                                    "- research_filter: when you start from a seed list and only want to follow promising links.\n"
                                    "Only used when research_mode is true."
                                ),
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
        self._detected_homonyms: List[str] = []

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

    def _is_valid_crawl_url(self, url: str) -> bool:
        """
        Determines if a URL is valid for crawling.
        Filters out static files, APIs, edit actions, etc.
        Special focus on Wikipedia, Reddit, and GitHub.
        """
        if not url or not isinstance(url, str):
            return False

        url_lower = url.lower()

        # Static file extensions to exclude
        static_extensions = (
            ".css",
            ".js",
            ".json",
            ".xml",
            ".rss",
            ".atom",
            ".png",
            ".jpg",
            ".jpeg",
            ".gif",
            ".svg",
            ".webp",
            ".ico",
            ".pdf",
            ".zip",
            ".tar",
            ".gz",
            ".mp4",
            ".mp3",
            ".webm",
            ".woff",
            ".woff2",
            ".ttf",
            ".eot",
            ".map",
            ".txt",
        )

        # Wikipedia-specific patterns to exclude
        wikipedia_exclude_patterns = (
            "/w/load.php",  # Dynamic load files
            "/w/api.php",  # Wikipedia API
            "/w/index.php?title=",  # Edit/history pages
            "action=edit",  # Edit
            "action=history",  # History
            "action=raw",  # Raw content
            "action=info",  # Page info
            "action=render",  # Render
            "action=submit",  # Form submission
            "action=delete",  # Deletion
            "action=protect",  # Protection
            "action=unprotect",  # Unprotection
            "action=move",  # Move page
            "special:recentchanges",  # Recent changes
            "special:whatlinkshere",  # What links here
            "special:export",  # Export
            "special:permalink",  # Permanent link
            "special:search",  # Search (results page)
            "special:userlogin",  # Login
            "special:createaccount",  # Create account
            "special:upload",  # Upload files
            "oldid=",  # Old versions
            "diff=",  # Differences between versions
            "printable=yes",  # Printable version
            "mobileaction=",  # Mobile version
            "veaction=edit",  # Visual editor
            "section=",  # Specific section
            "#",  # Internal anchors
            ".wikipedia.org/wiki/special:",  # Special in lowercase
            "simple.wikipedia.org",  # alternative domain
            "m.wikipedia.org",  # phone version
            "en.m.wikipedia.org",  # phone version
            "es.m.wikipedia.org",  # phone version
            "wikimedia.org",  # institutional site
        )

        # Reddit-specific patterns to exclude
        reddit_exclude_patterns = (
            "/r/",  # Subreddits (special processing)
            "/user/",  # User profiles
            "/comments/",  # Individual comments
            "/wiki/",  # Subreddit wikis
            "/message/",  # Messages
            "/submit/",  # Post submission
            "/login/",  # Login
            "/register/",  # Registration
            "/settings/",  # Settings
            "/prefs/",  # Preferences
            "/r/all/",  # All (too generic)
            "/r/popular/",  # Popular (generic)
            "?sort=",  # Sort parameters
            "?limit=",  # Limits
            "?after=",  # Pagination
            "?before=",  # Pagination
            "?context=",  # Context
            "?depth=",  # Depth
            "/api/",  # API endpoints
            "/embed/",  # Embedded
            "/saved/",  # Saved
            "/upvoted/",  # Upvoted
            "/downvoted/",  # Downvoted
            "/hidden/",  # Hidden
            "/gilded/",  # Gilded
            "/submitted/",  # User submissions
        )

        # GitHub-specific patterns to exclude
        github_exclude_patterns = (
            "/raw/",  # Raw content (unprocessed)
            "/blob/",  # Individual files (except README)
            "/commit/",  # Individual commits
            "/commits/",  # Commit history
            "/issues/",  # Issues
            "/pull/",  # Pull requests
            "/pulls/",  # Pull requests list
            "/actions/",  # GitHub Actions
            "/projects/",  # Projects
            "/wiki/",  # Wiki
            "/releases/",  # Releases page
            "/tags/",  # Tags
            "/branches/",  # Branches
            "/insights/",  # Insights
            "/settings/",  # Settings
            "/security/",  # Security
            "/labels/",  # Labels
            "/milestones/",  # Milestones
            "/discussions/",  # Discussions
            "/sponsor/",  # Sponsor
            "/account/",  # Account settings
            "/orgs/",  # Organization pages
            "/new/",  # New creation pages
            "/edit/",  # Edit pages
            "/delete/",  # Delete pages
            ".patch",  # Patch files
            ".diff",  # Diff files
            "?tab=",  # Tab parameters
            "#readme",  # README anchor (redundant)
        )

        # General patterns to exclude
        general_exclude_patterns = (
            "/login",
            "/logout",
            "/signup",
            "/register",
            "/cart",
            "/checkout",
            "/wishlist",
            "/admin",
            "/dashboard",
            "/settings",
            "/preferences",
            "/api/",
            "/graphql",
            "/rest/",
            "/rpc/",
            "/feed",
            "/rss",
            "/atom",
            "/xmlrpc.php",
            "/cdn-cgi/",
            "/wp-json/",
            "/wp-content/",
            "?share=",
            "?utm_",
            "?ref=",
            "?source=",
            "javascript:",
            "mailto:",
            "tel:",
            "data:",
            ".git/",
            ".svn/",
            ".env/",
            ".idea/",
            "/amp/",  # AMP pages
            "?format=amp",  # AMP format
            "/print/",  # Print versions
            "/mobile/",  # Mobile-specific
            "/embed/",  # Embedded content
            "/frame/",  # Frames
            "/iframe/",  # Iframes
            "?iframe=",  # Iframe parameter
            "/wiki/Wikipedia:",
            "/wiki/Portal:",
            "/wiki/Ayuda:",
            "/wiki/Especial:",
            "/wiki/Main_Page",
            "/wiki/Contents",
            "/wiki/Community_portal",
            "/wiki/Recent_changes",
            "/wiki/File:",
            "/wiki/Template:",
            "/wiki/Category:",
            "/wiki/Categor%C3%ADa",
            "/wiki/Help:",
            "/wiki/Special:",
        )

        # Check static extensions
        if any(url_lower.endswith(ext) for ext in static_extensions):
            if self.valves.DEBUG:
                logger.debug(f"Filtered: Static file extension - {url}")
            return False

        # Check Wikipedia patterns
        if "wikipedia.org" in url_lower:
            if any(pattern in url_lower for pattern in wikipedia_exclude_patterns):
                if self.valves.DEBUG:
                    logger.debug(f"Filtered: Wikipedia non-content page - {url}")
                return False
            # Verify it's a Wikipedia article (format /wiki/Article_Name)
            wiki_path = urlparse(url).path
            if not wiki_path.startswith("/wiki/") or len(wiki_path) < 7:
                if self.valves.DEBUG:
                    logger.debug(f"Filtered: Not a Wikipedia article - {url}")
                return False

        # Check Wikipedia namespaces (always excluded)
        if "wikipedia.org" in url_lower:
            wikipedia_namespaces = (
                "/wiki/wikipedia:",
                "/wiki/ayuda:",
                "/wiki/portal:",
                "/wiki/especial:",
                "/wiki/main_page",
                "/wiki/contents",
                "/wiki/categor%C3%ADa",
                "/wiki/help:",
                "/wiki/special:",
                "/wiki/file:",
                "/wiki/talk:",
                "/wiki/template:",
            )
            if any(pattern in url_lower for pattern in wikipedia_namespaces):
                if self.valves.DEBUG:
                    logger.debug(f"Filtered: Wikipedia namespace page - {url}")
                return False

        # Check Reddit patterns
        if "reddit.com" in url_lower or "redd.it" in url_lower:
            # Exclude URLs that are not main posts
            if any(pattern in url_lower for pattern in reddit_exclude_patterns):
                if self.valves.DEBUG:
                    logger.debug(f"Filtered: Reddit non-post page - {url}")
                return False
            # For Reddit, prefer text posts (self posts) and regular posts
            if "/comments/" not in url_lower:
                if self.valves.DEBUG:
                    logger.debug(f"Filtered: Not a Reddit post - {url}")
                return False

        # Check GitHub patterns
        if "github.com" in url_lower:
            if any(pattern in url_lower for pattern in github_exclude_patterns):
                if self.valves.DEBUG:
                    logger.debug(f"Filtered: GitHub non-content page - {url}")
                return False
            # For GitHub, prefer READMEs and markdown files
            if "/blob/" in url_lower:
                allowed_extensions = (".md", ".markdown", ".txt", ".rst", ".adoc")
                if not any(url_lower.endswith(ext) for ext in allowed_extensions):
                    if self.valves.DEBUG:
                        logger.debug(f"Filtered: GitHub non-readme file - {url}")
                    return False

        # Check general patterns
        if any(pattern in url_lower for pattern in general_exclude_patterns):
            if self.valves.DEBUG:
                logger.debug(f"Filtered: General exclude pattern - {url}")
            return False

        # Check URL length
        if len(url) > 500:
            if self.valves.DEBUG:
                logger.debug(f"Filtered: URL too long ({len(url)} chars)")
            return False

        # Check too many parameters
        parsed_url = urlparse(url)
        if parsed_url.query:
            query_params = parse_qs(parsed_url.query)
            if len(query_params) > 10:
                if self.valves.DEBUG:
                    logger.debug(
                        f"Filtered: Too many query parameters ({len(query_params)})"
                    )
                return False

        return True

    async def _has_keywords(self, url: str, keywords: List[str]) -> bool:
        """
        Fetch the page and check keywords in title and main content area.
        """
        # BYPASS for trusted domains
        trusted_domains = ["wikipedia.org", "wikimedia.org"]
        for domain in trusted_domains:
            if domain in url:
                if self.valves.DEBUG:
                    logger.debug(f"BYPASS keyword check for trusted domain: {url}")
                return True

        # Check for Wikipedia non-content pages
        if "wikipedia.org" in url:
            invalid_patterns = [
                "/w/load.php",
                "action=edit",
                "action=history",
                "special:recentchanges",
            ]
            if any(pattern in url.lower() for pattern in invalid_patterns):
                if self.valves.DEBUG:
                    logger.debug(f"SKIP: Wikipedia non-content page: {url}")
                return False

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            headers = {
                "User-Agent": self.valves.CRAWL4AI_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            }

            async with aiohttp.ClientSession(
                timeout=timeout, headers=headers
            ) as session:
                async with session.get(url, allow_redirects=True) as resp:
                    if resp.status >= 400:
                        return False

                    content = await resp.content.read(200 * 1024)
                    html = content.decode("utf-8", errors="ignore").lower()

                    title_match = re.search(
                        r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE
                    )
                    title_text = title_match.group(1) if title_match else ""

                    body_match = re.search(
                        r"<body[^>]*>(.*?)</body>", html, re.IGNORECASE | re.DOTALL
                    )
                    body_text = ""
                    if body_match:
                        body_clean = re.sub(
                            r"<script[^>]*>.*?</script>",
                            "",
                            body_match.group(1),
                            flags=re.DOTALL | re.IGNORECASE,
                        )
                        body_clean = re.sub(
                            r"<style[^>]*>.*?</style>",
                            "",
                            body_clean,
                            flags=re.DOTALL | re.IGNORECASE,
                        )
                        body_text = re.sub(r"<[^>]+>", " ", body_clean)
                        body_text = body_text[:5000]

                    searchable_text = f"{title_text} {body_text}"
                    matched = any(kw in searchable_text for kw in keywords)

                    if self.valves.DEBUG:
                        logger.debug(
                            f"GET {url} -> Keywords found: {matched} (Title: '{title_text[:100]}')"
                        )

                    return matched

        except asyncio.TimeoutError:
            if self.valves.DEBUG:
                logger.debug(f"GET timeout for {url}, passing through")
            return True
        except Exception as e:
            if self.valves.DEBUG:
                logger.debug(f"GET error for {url}: {str(e)}, passing through")
            return True

    def _is_html_url(self, url: str) -> bool:
        """Returns True if the URL likely points to an HTML page."""
        if not url or url.startswith(("javascript:", "mailto:", "tel:", "data:")):
            return False

        parsed = urlparse(url)
        path = parsed.path.rstrip("/")

        if not path or path == "/":
            return True

        last_segment = path.split("/")[-1]
        if "." not in last_segment:
            return True

        html_extensions = (
            ".html",
            ".htm",
            ".php",
            ".asp",
            ".aspx",
            ".jsp",
            ".jspx",
            ".do",
            ".action",
            ".cgi",
            ".pl",
            ".shtml",
            ".xhtml",
            ".cfm",
            ".phtml",
        )
        ext = "." + last_segment.split(".")[-1].lower()
        return ext in html_extensions

    async def _is_accessible_html(self, url: str) -> bool:
        """HTML check with forced acceptance for Wikipedia."""
        if "wikipedia.org" in url:
            if self.valves.DEBUG:
                logger.debug(f"BYPASS: Wikipedia URL automatically accepted: {url}")
            return True

        try:
            timeout = aiohttp.ClientTimeout(
                total=self.valves.PREFLIGHT_TIMEOUT,
                sock_read=self.valves.PREFLIGHT_TIMEOUT // 2,
            )
            headers = {
                "User-Agent": self.valves.CRAWL4AI_USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
            }

            async with aiohttp.ClientSession(
                timeout=timeout, headers=headers
            ) as session:
                async with session.get(url, allow_redirects=True) as resp:
                    if resp.status >= 400:
                        if self.valves.DEBUG:
                            logger.debug(f"GET {url} -> status {resp.status}")
                        return False

                    content_type = resp.headers.get("Content-Type", "").lower()
                    if self.valves.DEBUG:
                        logger.debug(f"GET {url} -> Content-Type: {content_type}")

                    if (
                        "text/html" in content_type
                        or "application/xhtml+xml" in content_type
                    ):
                        return True

                    chunk = b""
                    async for data in resp.content.iter_chunked(1024):
                        chunk += data
                        if len(chunk) >= 10240:
                            break

                    chunk_lower = chunk.lower()
                    if (
                        b"<html" in chunk_lower
                        or b"<!doctype" in chunk_lower
                        or b"<?xml" in chunk_lower
                    ):
                        return True

                    return False

        except Exception as e:
            if self.valves.DEBUG:
                logger.debug(f"Error checking {url}: {str(e)}")
            return False

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

    # endregion

    # region ── URL Validation Pipeline ────────────────────────────────────────

    async def _validate_url_pipeline(
        self,
        urls: List[str],
        query: Optional[str] = None,
        check_accessibility: bool = True,
        check_keywords: bool = True,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> List[str]:
        """
        Centralized URL validation pipeline used by both regular crawl
        and research crawl strategies. Avoids duplicating the same
        _is_valid_crawl_url / _is_accessible_html / _has_keywords
        call sequence in multiple places.
        """
        original_count = len(urls)

        # Stage 1: Filter invalid URLs
        urls = [url for url in urls if self._is_valid_crawl_url(url)]
        if self.valves.DEBUG and original_count != len(urls):
            logger.info(
                f"Validation stage 1: filtered out "
                f"{original_count - len(urls)} invalid URLs"
            )

        if not urls:
            if __event_emitter__ and self.valves.MORE_STATUS:
                if self.valves.DEBUG:
                    logger.debug("No valid URLs to crawl after filtering")
            return []

        # Stage 2: Check HTML accessibility
        if check_accessibility:
            if self.valves.DEBUG:
                logger.info(
                    f"Validation stage 2: checking accessibility for "
                    f"{len(urls)} URLs..."
                )
            tasks = [self._is_accessible_html(url) for url in urls]
            results = await asyncio.gather(*tasks)
            urls = [url for url, ok in zip(urls, results) if ok]
            if self.valves.DEBUG:
                logger.info(f"Validation stage 2: {len(urls)} accessible HTML URLs")

        if not urls:
            return []

        # Stage 3: Check keywords in content
        if check_keywords and query:
            # Strip stopwords and short words to get meaningful anchor terms
            stopwords = {
                "de",
                "la",
                "el",
                "los",
                "las",
                "del",
                "en",
                "y",
                "o",
                "un",
                "una",
                "the",
                "of",
                "and",
                "or",
                "a",
                "an",
                "is",
                "que",
                "se",
                "su",
                "con",
                "por",
                "para",
                "al",
                "lo",
            }
            raw_keywords = [
                re.sub(r"[^\w]", "", kw).lower()
                for kw in query.split()
                if re.sub(r"[^\w]", "", kw)
            ]
            preflight_keywords = [
                kw for kw in raw_keywords if len(kw) > 3 and kw not in stopwords
            ]

            # Anchor words: the longest terms are the most semantically specific
            anchor_keywords = sorted(set(preflight_keywords), key=len, reverse=True)[:2]

            if preflight_keywords:
                if self.valves.DEBUG:
                    logger.info(
                        f"Validation stage 3: checking keywords {preflight_keywords} "
                        f"(anchors: {anchor_keywords}) for {len(urls)} URLs..."
                    )

                # General keyword check
                tasks = [self._has_keywords(url, preflight_keywords) for url in urls]
                general_results = await asyncio.gather(*tasks)

                # Anchor keyword check (stricter: must contain the core subject)
                if anchor_keywords:
                    anchor_tasks = [
                        self._has_keywords(url, anchor_keywords) for url in urls
                    ]
                    anchor_results = await asyncio.gather(*anchor_tasks)
                    urls = [
                        url
                        for url, general_ok, anchor_ok in zip(
                            urls, general_results, anchor_results
                        )
                        if general_ok and anchor_ok
                    ]
                else:
                    urls = [url for url, ok in zip(urls, general_results) if ok]

                if self.valves.DEBUG:
                    logger.info(
                        f"Validation stage 3: {len(urls)} URLs after keyword+anchor check"
                    )

        return urls

    # endregion

    # region ── LLM Query Expansion & URL Filter ───────────────────────────────

    async def _expand_query_with_llm(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> List[str]:
        """Generate related search terms using LLM.
        Uses EXPANSION_LLM_PROVIDER valve (falls back to LLM_PROVIDER).
        """
        if not self.valves.USE_QUERY_EXPANSION:
            return [query]

        if __event_emitter__ and self.valves.MORE_STATUS:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Generating related search terms for '{query}'...",
                        "done": False,
                    },
                }
            )

        prompt = f"""You are a search query expansion expert. Your task is to generate search terms that preserve the EXACT intended meaning of the original query, including any disambiguation.
    
    Original query: "{query}"
    
    **Step 1 — Identify the core concept:**
    Before generating any terms, explicitly state:
    - What is the EXACT meaning of the key subject in this query? (not a category, the specific thing)
    - Are there homonyms or other meanings for this word that must be AVOIDED?
    
    **Step 2 — Generate {self.valves.MAX_EXPANDED_QUERIES} search terms following these rules:**
    - Every term MUST be about the identical concept identified in Step 1
    - Include: synonyms, scientific names, alternative phrasings, specific subtopics
    - Each term must contain enough context words to disambiguate from homonyms
      (e.g., if the subject has multiple meanings, add a disambiguating word)
    - Keep terms under 10 words each
    - NEVER generate terms that are a broader category (e.g., "fruit" alone)
    - NEVER generate terms about a different meaning of the same word
    
    **Output format:**
    First output your Step 1 analysis as comments, then output ONLY a JSON list:
    // Core concept: [your analysis here]
    // Homonyms to avoid: [comma-separated list, e.g. "fresadora, fresa dental, fresa color"]
    ["term 1", "term 2", ...]
    
    """

        base_url = self.valves.LLM_BASE_URL.rstrip("/")
        provider = self.valves.EXPANSION_LLM_PROVIDER or self.valves.LLM_PROVIDER
        api_token = (
            self.valves.LLM_API_TOKEN.strip()
            if self.valves.LLM_API_TOKEN and self.valves.LLM_API_TOKEN.strip()
            else None
        )

        valve_model = provider
        if "/" in valve_model:
            valve_model = valve_model.split("/", 1)[1]

        is_ollama = "ollama" in base_url.lower() or ":11434" in base_url

        try:
            if is_ollama:
                ollama_url = f"{base_url}/api/generate"
                payload = {
                    "model": valve_model,
                    "prompt": prompt,
                    "system": "You are a precise search query expander. Respond only with the requested format: comments then JSON list.",
                    "stream": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 700,
                    },
                }
                response = requests.post(ollama_url, json=payload, timeout=30)

                if response.status_code != 200:
                    logger.error(
                        f"Ollama expansion error {response.status_code}: {response.text[:200]}"
                    )
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "⚠️ Query expansion failed. Using original query.",
                                    "done": False,
                                },
                            }
                        )
                    return [query]

                result = response.json()
                content = result.get("response", "")
            else:
                headers = {"Content-Type": "application/json"}
                if api_token:
                    headers["Authorization"] = f"Bearer {api_token}"

                payload = {
                    "model": valve_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a precise search query expander. Respond only with the requested format: comments then JSON list.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.3,
                    "max_tokens": 700,
                }

                response = requests.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30,
                )

                if response.status_code != 200:
                    logger.error(
                        f"Expansion error {response.status_code}: {response.text[:200]}"
                    )
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "⚠️ Query expansion failed. Using original query.",
                                    "done": False,
                                },
                            }
                        )
                    return [query]

                result = response.json()
                content = (
                    result.get("choices", [{}])[0].get("message", {}).get("content", "")
                )

            import json

            # Extract homonyms from Step 1 comment and store for use by URL filter
            self._detected_homonyms = []
            homonym_match = re.search(
                r"//\s*Homonyms to avoid:\s*(.+)", content, re.IGNORECASE
            )
            if homonym_match:
                raw = homonym_match.group(1).strip()
                self._detected_homonyms = [
                    h.strip().lower()
                    for h in re.split(r"[,;]", raw)
                    if h.strip()
                    and h.strip().lower() not in ("none", "ninguno", "n/a", "-")
                ]
                if self.valves.DEBUG:
                    logger.info(
                        f"Detected homonyms from expansion: {self._detected_homonyms}"
                    )

            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                related_queries = json.loads(json_match.group())
                if isinstance(related_queries, list):
                    related_queries = [
                        str(q).strip() for q in related_queries if q and str(q).strip()
                    ]
                    all_queries = [query]
                    for q in related_queries[: self.valves.MAX_EXPANDED_QUERIES]:
                        if q and q.lower() != query.lower() and q not in all_queries:
                            all_queries.append(q)

                    if self.valves.DEBUG:
                        logger.info(f"Query expansion: {query} -> {all_queries}")

                    if __event_emitter__ and self.valves.MORE_STATUS:
                        queries_display = "\n".join(
                            [f"  • {q}" for q in all_queries[1:]]
                        )
                        if len(all_queries) > 1:
                            await __event_emitter__(
                                {
                                    "type": "status",
                                    "data": {
                                        "description": f"🔍 Expanded search terms:\n{queries_display}",
                                        "done": False,
                                    },
                                }
                            )
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": f"Will search using {len(all_queries)} terms (including original)",
                                    "done": False,
                                },
                            }
                        )

                    return all_queries

            logger.warning("Query expansion: LLM response did not contain valid JSON")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "⚠️ Could not parse expanded queries. Using original query.",
                            "done": False,
                        },
                    }
                )

        except Exception as e:
            logger.error(f"Query expansion error: {e}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "Query expansion failed, using original query.",
                            "done": False,
                        },
                    }
                )

        return [query]

    async def _search_all_queries(
        self,
        queries: List[str],
        __event_emitter__: Callable[[dict], Any] = None,
        __user__: Optional[dict] = None,
    ) -> List[str]:
        """Perform searches for multiple queries and aggregate unique URLs."""
        all_urls = []

        # Unified search start message
        if __event_emitter__ and self.valves.MORE_STATUS:
            engine_names = []
            if self.valves.USE_NATIVE_SEARCH:
                engine_names.append("Native")
            if self.valves.SEARCH_WITH_SEARXNG:
                engine_names.append("SearXNG")
            engines_str = " & ".join(engine_names)
            if len(queries) > 1:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"📡 Searching {len(queries)} terms with {engines_str}...",
                            "done": False,
                        },
                    }
                )
            else:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"📡 Searching with {engines_str}...",
                            "done": False,
                        },
                    }
                )

        for idx, query in enumerate(queries, 1):
            if __event_emitter__ and self.valves.MORE_STATUS:
                icon = "🎯" if idx == 1 else "🔄"
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f'{icon} Searching ({idx}/{len(queries)}): "{query}"',
                            "done": False,
                        },
                    }
                )

            query_urls = []

            if self.valves.USE_NATIVE_SEARCH:
                native_urls = await self._search_native(
                    query, __event_emitter__, __user__
                )
                query_urls.extend(native_urls)

            if self.valves.SEARCH_WITH_SEARXNG:
                searxng_urls = await self._search_searxng(query, __event_emitter__)
                max_results = (
                    self.user_valves.SEARXNG_MAX_RESULTS
                    or self.valves.SEARXNG_MAX_RESULTS
                )
                query_urls.extend(searxng_urls[:max_results])

            if __event_emitter__ and self.valves.MORE_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"  → Found {len(query_urls)} URLs",
                            "done": False,
                        },
                    }
                )

            for url in query_urls:
                if url not in all_urls:
                    all_urls.append(url)

        if self.valves.DEBUG:
            logger.info(
                f"Search across {len(queries)} queries returned {len(all_urls)} unique URLs"
            )

        if __event_emitter__ and self.valves.MORE_STATUS:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"📊 Total unique URLs found: {len(all_urls)}",
                        "done": False,
                    },
                }
            )

        return all_urls

    async def _filter_urls_with_llm(
        self,
        urls: List[str],
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> List[str]:
        """Filter URLs using LLM for semantic relevance.
        Uses FILTER_LLM_PROVIDER valve (falls back to LLM_PROVIDER).
        """
        if not self.valves.USE_LLM_URL_FILTER or not urls:
            return urls

        if __event_emitter__ and self.valves.MORE_STATUS:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"Using LLM to filter {len(urls)} URLs for relevance...",
                        "done": False,
                    },
                }
            )

        # pre-filter - reject homonimn slugs
        detected = getattr(self, "_detected_homonyms", [])
        if detected:
            pre_filtered = []
            for url in urls:
                slug = urlparse(url).path.lower()
                slug_clean = slug.replace("_", " ").replace("-", " ")
                rejected = False
                for homonym in detected:
                    homonym_clean = homonym.replace("_", " ").replace("-", " ")
                    # First try full phrase match
                    if re.search(rf"\b{re.escape(homonym_clean)}\b", slug_clean):
                        rejected = True
                        break
                    # If it fails, check that ALL significant tokens of the homonym are in the slug
                    tokens = [t for t in homonym_clean.split() if len(t) > 4]
                    if tokens and all(t in slug_clean for t in tokens):
                        if self.valves.DEBUG:
                            logger.info(
                                f"Pre-filter (token match): rejected {url} (homonym: '{homonym}')"
                            )
                        rejected = True
                        break
                if not rejected:
                    pre_filtered.append(url)

            removed = len(urls) - len(pre_filtered)
            if removed > 0:
                if self.valves.DEBUG:
                    logger.info(
                        f"Pre-filter removed {removed} URLs by homonym slug match"
                    )
                if __event_emitter__ and self.valves.MORE_STATUS:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"Pre-filter: removed {removed} URL{'s' if removed != 1 else ''} by homonym match.",
                                "done": False,
                            },
                        }
                    )
            urls = pre_filtered

        if not urls:
            return []

        url_titles = {}

        async def fetch_title(url: str) -> tuple[str, str]:
            try:
                timeout = aiohttp.ClientTimeout(total=5)
                headers = {"User-Agent": self.valves.CRAWL4AI_USER_AGENT}
                async with aiohttp.ClientSession(
                    timeout=timeout, headers=headers
                ) as session:
                    async with session.get(url, allow_redirects=True) as resp:
                        if resp.status == 200:
                            content = await resp.content.read(50 * 1024)
                            html = content.decode("utf-8", errors="ignore").lower()
                            title_match = re.search(
                                r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE
                            )
                            if title_match:
                                title = title_match.group(1).strip()[:200]
                                return url, title
            except Exception as e:
                if self.valves.DEBUG:
                    logger.debug(f"Could not fetch title for {url}: {e}")
            return url, ""

        tasks = [fetch_title(url) for url in urls]
        results = await asyncio.gather(*tasks)
        for url, title in results:
            url_titles[url] = title

        base_url = self.valves.LLM_BASE_URL.rstrip("/")
        api_token = (
            self.valves.LLM_API_TOKEN.strip()
            if self.valves.LLM_API_TOKEN and self.valves.LLM_API_TOKEN.strip()
            else None
        )
        is_ollama = "ollama" in base_url.lower() or ":11434" in base_url

        used_model = self.valves.FILTER_LLM_PROVIDER or self.valves.LLM_PROVIDER
        if "/" in used_model:
            used_model = used_model.split("/", 1)[1]

        # Build homonym context from expansion step if available
        homonym_context = ""
        if detected:
            homonym_list = ", ".join(detected)
            homonym_context = f"\nPre-detected homonyms (MUST REJECT pages about these): {homonym_list}\n"

        prompt = f"""You are a semantic URL filter. Your task is to determine if a URL is about the EXACT SAME CONCEPT as the user's query.
        
        USER QUERY: "{query}"
        
        == DISAMBIGUATION STEP (REQUIRED - do this before evaluating any URL) ==
        
        The query may contain ambiguous words. Based on the FULL CONTEXT of the query:
        
        1. Identify the INTENDED meaning of key terms
        2. List alternative meanings (homonyms) that should be REJECTED
        3. State the core concept in a clear, unambiguous phrase
        
        Example:
        Query: "clasificacion de tipos de fresa (fruta)"
        → Intended: Fruit (strawberry) classification/taxonomy
        → Reject: Dental burs, milling cutters, strawberry plant morphology only (without fruit classification)
        
        == EVALUATION RULES ==
        
        For each URL, apply this reasoning:
        - Does the page content match the INTENDED meaning identified above?
        - Does it contain terms from the REJECT list? → REJECT if yes
        - If the query explicitly includes a disambiguator like "(fruta)", "(fruit)", "(dental)", etc., RESPECT IT
        
        REJECT when:
        - The URL is about a homonym/alternative meaning
        - The URL is a disambiguation page, category, portal, or "List of..."
        - You are uncertain (false negative is better than crawling irrelevant content)
        
        == OUTPUT ==
        
        Return ONLY a JSON list:
        [{{"index":1,"decision":"KEEP"}},{{"index":2,"decision":"REJECT"}}]
        
        URLs to evaluate:
        """
        for idx, url in enumerate(urls, 1):
            title = url_titles.get(url, "No title available")
            prompt += f"{idx}. URL: {url}\n   Title: {title}\n"

        try:
            if is_ollama:
                ollama_url = f"{base_url}/api/generate"
                payload = {
                    "model": used_model,
                    "prompt": prompt,
                    "system": "You are a precise URL relevance filter. Respond only with the requested JSON format.",
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 2000,
                    },
                }
                response = requests.post(ollama_url, json=payload, timeout=30)

                if response.status_code != 200:
                    logger.error(f"LLM URL filter Ollama error: {response.status_code}")
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "LLM filter failed, continuing with all URLs.",
                                    "done": False,
                                },
                            }
                        )
                    return urls

                result = response.json()
                content = result.get("response", "")
            else:
                headers = {"Content-Type": "application/json"}
                if api_token:
                    headers["Authorization"] = f"Bearer {api_token}"

                payload = {
                    "model": used_model,
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are a precise URL relevance filter. Respond only with the requested JSON format.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.1,
                    "max_tokens": 2000,
                }

                response = requests.post(
                    f"{base_url}/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=30,
                )

                if response.status_code != 200:
                    logger.error(f"LLM URL filter error: {response.status_code}")
                    if __event_emitter__:
                        await __event_emitter__(
                            {
                                "type": "status",
                                "data": {
                                    "description": "LLM filter failed, continuing with all URLs.",
                                    "done": False,
                                },
                            }
                        )
                    return urls

                result = response.json()
                content = (
                    result.get("choices", [{}])[0].get("message", {}).get("content", "")
                )

            import json

            json_match = re.search(r"\[.*\]", content, re.DOTALL)
            if json_match:
                decisions = json.loads(json_match.group())
                keep_indices = {
                    item["index"] - 1
                    for item in decisions
                    if item.get("decision") == "KEEP"
                }
                filtered_urls = [urls[i] for i in range(len(urls)) if i in keep_indices]

                if self.valves.DEBUG:
                    logger.info(
                        f"LLM URL filter: kept {len(filtered_urls)}/{len(urls)} URLs"
                    )

                if __event_emitter__ and self.valves.MORE_STATUS:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"LLM filter: keeping {len(filtered_urls)} relevant URLs, rejecting {len(urls) - len(filtered_urls)}.",
                                "done": False,
                            },
                        }
                    )

                return filtered_urls

        except Exception as e:
            logger.error(f"LLM URL filter error: {e}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": "LLM filter failed, continuing with all URLs.",
                            "done": False,
                        },
                    }
                )

        return urls

    # endregion

    # region ── Image Validation ───────────────────────────────────────────────

    async def _validate_image_url(self, url: str) -> bool:
        """Validate if an image URL is accessible."""
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
                        return False
                    content_type = response.headers.get("Content-Type", "").lower()
                    if not content_type.startswith("image/"):
                        return False
                    return True
        except Exception:
            return False

    async def _validate_images_batch(self, urls: List[str]) -> List[str]:
        """Validate multiple image URLs concurrently."""
        tasks = [self._validate_image_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return [url for url, is_valid in zip(urls, results) if is_valid]

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
        """Search using OpenWebUI's native web search."""
        if not self.valves.USE_NATIVE_SEARCH:
            return []

        if not NATIVE_SEARCH_AVAILABLE:
            logger.warning("Native search not available")
            return []

        if __user__ is None:
            logger.error("User information required for native search")
            return []

        try:
            user = await Users.get_user_by_id(__user__["id"])
            if user is None:
                logger.error("User not found")
                return []

            form = SearchForm.model_validate({"queries": [query]})
            result = await process_web_search(
                request=Request(scope={"type": "http", "app": app}),
                form_data=form,
                user=user,
            )

            urls = [
                item.get("link") for item in result.get("items", []) if item.get("link")
            ]

            if self.valves.DEBUG:
                logger.info(f"Native search for '{query}' returned {len(urls)} URLs")

            return urls

        except Exception as e:
            logger.error(f"Error in native search: {str(e)}")
            return []

    async def _search_searxng(
        self,
        query: str,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> List[str]:
        """Search SearXNG and return a list of URLs."""
        if not self.valves.SEARCH_WITH_SEARXNG:
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

            return urls

        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching SearXNG: {str(e)}")
            if __event_emitter__:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"⚠️ Error in SearXNG search: {str(e)[:100]}",
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
        Main entry point for web search and crawl.
        """
        logger.info(f"Starting search and crawl for '{query}'")

        gathered_urls = []
        user_provided_urls = []
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
                if self._is_html_url(url) and url not in gathered_urls:
                    gathered_urls.append(url)
                    user_provided_urls.append(url)

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

        # Expansion and filtering now use dedicated valves (no auto-detection)
        search_queries = await self._expand_query_with_llm(query, __event_emitter__)
        search_urls = await self._search_all_queries(
            search_queries, __event_emitter__, __user__
        )

        for url in search_urls:
            if url not in gathered_urls:
                gathered_urls.append(url)

        if max_results and max_results > 0 and len(gathered_urls) > max_results:
            gathered_urls = gathered_urls[:max_results]

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
            return f"No URLs found to crawl for the query: {query}."

        if self.valves.USE_LLM_URL_FILTER and len(gathered_urls) > 0:
            if __event_emitter__ and self.valves.MORE_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"🧠 AI Semantic Filtering analyzing {len(gathered_urls)} URLs...",
                            "done": False,
                        },
                    }
                )
                await asyncio.sleep(0.2)

            before_filter_count = len(gathered_urls)
            gathered_urls = await self._filter_urls_with_llm(
                gathered_urls, query, __event_emitter__
            )

            if __event_emitter__ and self.valves.MORE_STATUS:
                removed_count = before_filter_count - len(gathered_urls)
                if removed_count > 0:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": f"✅ AI filter: removed {removed_count} irrelevant URL{'s' if removed_count != 1 else ''}. {len(gathered_urls)} remain.",
                                "done": False,
                            },
                        }
                    )
                else:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "✅ AI filter: all URLs appear relevant.",
                                "done": False,
                            },
                        }
                    )
                await asyncio.sleep(0.3)

            if not gathered_urls:
                if __event_emitter__:
                    await __event_emitter__(
                        {
                            "type": "status",
                            "data": {
                                "description": "❌ AI filter removed all URLs.",
                                "done": True,
                            },
                        }
                    )
                return f"No relevant URLs were found for the query: {query}."

        # Separamos las URLs proporcionadas por el usuario de las de búsqueda
        search_only_urls = [
            url for url in gathered_urls if url not in user_provided_urls
        ]

        # Unimos las URLs del usuario primero, luego las de búsqueda (en orden original)
        gathered_urls = user_provided_urls + search_only_urls

        max_urls = self.user_valves.CRAWL4AI_MAX_URLS or self.valves.CRAWL4AI_MAX_URLS
        if len(gathered_urls) > max_urls:
            if self.valves.DEBUG:
                logger.info(f"Limiting URLs from {len(gathered_urls)} to {max_urls}")
            if __event_emitter__ and self.valves.MORE_STATUS:
                await __event_emitter__(
                    {
                        "type": "status",
                        "data": {
                            "description": f"✂️ Limitando a {max_urls} URLs (de {len(gathered_urls)} total). Priorizando URLs proporcionadas por el usuario.",
                            "done": False,
                        },
                    }
                )
                await asyncio.sleep(0.3)
            gathered_urls = gathered_urls[:max_urls]

        # Determine research mode and crawl strategy
        if research_crawl_mode and research_crawl_mode in [
            "pseudo_adaptive",
            "llm_guided",
            "bfs_deep",
            "research_filter",
        ]:
            effective_research_mode = True
            effective_crawl_mode = research_crawl_mode
            if self.valves.DEBUG:
                logger.info(
                    f"Research mode activated via research_crawl_mode parameter: {research_crawl_mode}"
                )
        else:
            effective_research_mode = research_mode or self.user_valves.RESEARCH_MODE
            effective_crawl_mode = (
                research_crawl_mode or self.user_valves.RESEARCH_CRAWL_MODE
            )
            if self.valves.DEBUG:
                logger.info(
                    f"Research mode: {effective_research_mode}, mode: {effective_crawl_mode}"
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

        if __event_emitter__ and self.valves.MORE_STATUS:
            await __event_emitter__(
                {
                    "type": "status",
                    "data": {
                        "description": f"🔍 Processing {len(gathered_urls)} most relevant results...",
                        "done": False,
                    },
                }
            )

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
                max_tokens=self.valves.CRAWL4AI_MAX_TOKENS,
                max_urls=max_urls,
                __event_emitter__=__event_emitter__,
            )

            if "content" in research_result:
                crawl_results.extend(research_result["content"])
                if self.valves.DEBUG:
                    logger.info(
                        f"Research mode added {len(research_result['content'])} content items"
                    )
            if "images" in research_result:
                image_list.extend(research_result["images"])
            if "videos" in research_result:
                video_list.extend(research_result["videos"])

        else:
            # Validate ALL URLs once before batching
            gathered_urls = await self._validate_url_pipeline(
                gathered_urls,
                query,
                check_keywords=True,
                __event_emitter__=__event_emitter__,
            )

            for i in range(0, len(gathered_urls), self.valves.CRAWL4AI_BATCH):
                batch = gathered_urls[i : i + self.valves.CRAWL4AI_BATCH]
                try:
                    crawled_batch = await self._crawl_url(
                        urls=batch,
                        query=query,
                        skip_validation=True,
                        __event_emitter__=__event_emitter__,
                    )

                    if self.valves.DEBUG:
                        logger.info(
                            f"Found {len(crawled_batch.get('content', []))} content, "
                            f"{len(crawled_batch.get('images', []))} images, "
                            f"{len(crawled_batch.get('videos', []))} videos."
                        )

                    for img_url in crawled_batch.get("images", []):
                        parsed_image = urlparse(img_url)
                        base_image_url = f"{parsed_image.scheme}://{parsed_image.netloc}{parsed_image.path}"
                        if base_image_url in seen_images:
                            continue
                        seen_images.add(base_image_url)
                        thumbnail_url = (
                            f"https://images.weserv.nl/?url={quote(img_url)}"
                            f"&w={thumbnail_size}&h={thumbnail_size}&fit=inside"
                        )
                        if await self._validate_image_url(
                            img_url
                        ) and await self._validate_image_url(thumbnail_url):
                            image_list.append(img_url)

                    for vid_url in crawled_batch.get("videos", []):
                        parsed_video = urlparse(vid_url)
                        base_video_url = f"{parsed_video.scheme}://{parsed_video.netloc}{parsed_video.path}"
                        if base_video_url in seen_videos:
                            continue
                        seen_videos.add(base_video_url)
                        video_list.append(vid_url)

                    data_list = crawled_batch.get("content", [])
                    normalized_data_list = self._normalize_content(data_list)

                    if normalized_data_list:
                        content_str = orjson.dumps(normalized_data_list).decode("utf-8")
                        page_tokens = await self._count_tokens(content_str)

                        if (
                            self.valves.CRAWL4AI_MAX_TOKENS > 0
                            and page_tokens > self.valves.CRAWL4AI_MAX_TOKENS
                        ):
                            content_str = await self._truncate_content(
                                content_str, self.valves.CRAWL4AI_MAX_TOKENS
                            )
                            try:
                                normalized_data_list = orjson.loads(
                                    content_str.replace(
                                        "\n\n[Content truncated due to length...]", ""
                                    )
                                )
                            except Exception:
                                pass
                            page_tokens = self.valves.CRAWL4AI_MAX_TOKENS

                        if (
                            self.valves.CRAWL4AI_MAX_TOKENS > 0
                            and total_tokens + page_tokens
                            > self.valves.CRAWL4AI_MAX_TOKENS
                        ):
                            logger.warning(
                                f"Reached token limit. Skipping remaining pages."
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
                        crawl_results.extend(normalized_data_list)

                    batch_count += 1

                except Exception as e:
                    logger.error(
                        f"An unexpected error occurred: {str(e)}\n{traceback.format_exc()}"
                    )

        crawl_results = self._normalize_content(crawl_results)

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

    # region ── Core Crawler ───────────────────────────────────────────────────

    async def _crawl_url(
        self,
        urls: Union[list, str],
        query: Optional[str] = None,
        extract_links: bool = False,
        skip_validation: bool = False,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """
        Internal function to crawl URLs and extract content.
        """
        if isinstance(urls, str):
            urls = [urls]

        urls = [
            url if url.startswith(("http://", "https://")) else f"https://{url}"
            for url in urls
        ]

        if not skip_validation:
            urls = await self._validate_url_pipeline(
                urls, query, __event_emitter__=__event_emitter__
            )

        if not urls:
            return {"content": [], "images": [], "videos": [], "links": []}

        base_url = self.valves.CRAWL4AI_BASE_URL.rstrip("/")
        endpoint = f"{base_url}/crawl"

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

        headers = {"Content-Type": "application/json"}
        payload = {
            "urls": urls,
            "browser_config": browser_config.dump(),
            "crawler_config": crawler_config.dump(),
        }

        timeout = self.valves.CRAWL4AI_TIMEOUT * len(urls) + 60
        try:
            response = await anyio.to_thread.run_sync(
                lambda: requests.post(
                    endpoint, json=payload, headers=headers, timeout=timeout
                )
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
                        if not self._is_valid_crawl_url(match):
                            continue
                        if match.startswith("http") and self._is_html_url(match):
                            all_links.append(match)

                await __event_emitter__(
                    {"type": "files", "data": {"files": image_list + video_list}}
                )

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
                            and content_item.get("error") is False
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

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            error_msg = f"Cannot connect to Crawl4AI at {endpoint}: {str(e)}"
            logger.error(error_msg)
            return {"error": error_msg, "content": [], "images": [], "videos": []}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}\n{traceback.format_exc()}")
            return {"content": [], "images": [], "videos": [], "links": []}

    # endregion

    # region ── Research Crawl Router ──────────────────────────────────────────

    async def _research_crawl(
        self,
        urls: List[str],
        query: str,
        mode: str = "pseudo_adaptive",
        max_tokens: int = 0,
        max_urls: Optional[int] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """Route to the appropriate research crawling strategy."""
        if max_urls is None:
            max_urls = (
                self.user_valves.CRAWL4AI_MAX_URLS or self.valves.CRAWL4AI_MAX_URLS
            )

        if mode == ResearchCrawlMode.PSEUDO_ADAPTIVE:
            return await self._pseudo_adaptive_crawl(
                urls, query, max_tokens, max_urls, __event_emitter__
            )
        elif mode == ResearchCrawlMode.LLM_GUIDED:
            return await self._llm_guided_crawl(
                urls, query, max_tokens, max_urls, __event_emitter__
            )
        elif mode == ResearchCrawlMode.BFS_DEEP:
            return await self._bfs_deep_crawl(
                urls, query, max_tokens, max_urls, __event_emitter__
            )
        elif mode == ResearchCrawlMode.RESEARCH_FILTER:
            return await self._research_filter_crawl(
                urls, query, max_tokens, max_urls, __event_emitter__
            )
        else:
            logger.warning(
                f"Unknown research crawl mode: {mode}, defaulting to pseudo_adaptive"
            )
            return await self._pseudo_adaptive_crawl(
                urls, query, max_tokens, max_urls, __event_emitter__
            )

    # endregion

    # region ── Research Crawl Strategies ─────────────────────────────────────

    async def _pseudo_adaptive_crawl(
        self,
        start_urls: List[str],
        query: str,
        max_tokens: int = 0,
        max_urls: Optional[int] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """Pseudo‑adaptive crawl: scores URLs based on keyword match."""
        from collections import deque

        if max_urls is None:
            max_urls = (
                self.user_valves.CRAWL4AI_MAX_URLS or self.valves.CRAWL4AI_MAX_URLS
            )

        max_pages = max_urls
        max_depth = self.user_valves.RESEARCH_MAX_DEPTH
        batch_size = self.user_valves.RESEARCH_BATCH_SIZE
        include_external = self.user_valves.RESEARCH_INCLUDE_EXTERNAL

        keywords = query.lower().split()
        crawled_pages = set()
        crawled_results = []
        all_images = []
        all_videos = []
        total_tokens = 0

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

                # Validate URL before crawling
                validated = await self._validate_url_pipeline(
                    [url],
                    query,
                    check_keywords=False,
                    __event_emitter__=__event_emitter__,
                )
                if not validated:
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
                    extract_links=True,
                    skip_validation=True,
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

                        if max_tokens > 0 and total_tokens + page_tokens > max_tokens:
                            logger.warning(
                                f"Token limit reached. Stopping research crawl."
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
                    for link in result.get("links", []):
                        # Filter invalid URLs before adding to queue
                        if not self._is_valid_crawl_url(link):
                            if self.valves.DEBUG:
                                logger.debug(
                                    f"Skipping invalid URL in research mode: {link[:100]}"
                                )
                            continue
                        if link in crawled_pages:
                            continue
                        parsed_link = urlparse(link)
                        parsed_url = urlparse(url)
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
        max_urls: Optional[int] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """LLM‑guided crawl: uses an LLM to select which links to follow next."""
        if max_urls is None:
            max_urls = (
                self.user_valves.CRAWL4AI_MAX_URLS or self.valves.CRAWL4AI_MAX_URLS
            )

        max_pages = max_urls
        include_external = self.user_valves.RESEARCH_INCLUDE_EXTERNAL

        crawled_pages = set()
        crawled_results = []
        all_images = []
        all_videos = []
        total_tokens = 0

        urls_to_process = list(start_urls)

        while (
            urls_to_process
            and len(crawled_pages) < max_pages
            and (max_tokens == 0 or total_tokens < max_tokens)
        ):
            current_url = urls_to_process.pop(0)

            if current_url in crawled_pages:
                continue

            # Validate URL before crawling
            validated = await self._validate_url_pipeline(
                [current_url],
                query,
                check_keywords=False,
                __event_emitter__=__event_emitter__,
            )
            if not validated:
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
                extract_links=True,
                skip_validation=True,
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

                    if max_tokens > 0 and total_tokens + page_tokens > max_tokens:
                        logger.warning(f"Token limit reached. Stopping research crawl.")
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

            # Filter invalid URLs before processing
            discovered_links = [
                link for link in discovered_links if self._is_valid_crawl_url(link)
            ]

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
        max_urls: Optional[int] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """Breadth‑first deep crawl: explores pages layer by layer."""
        from collections import deque

        if max_urls is None:
            max_urls = (
                self.user_valves.CRAWL4AI_MAX_URLS or self.valves.CRAWL4AI_MAX_URLS
            )

        max_pages = max_urls
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

                # Validate URL before crawling
                validated = await self._validate_url_pipeline(
                    [url],
                    query,
                    check_keywords=False,
                    __event_emitter__=__event_emitter__,
                )
                if not validated:
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
                    extract_links=True,
                    skip_validation=True,  # ← already validated above by _validate_url_pipeline
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

                        if max_tokens > 0 and total_tokens + page_tokens > max_tokens:
                            logger.warning(
                                f"Token limit reached. Stopping research crawl."
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
                        # Filter invalid URLs before adding to queue
                        if not self._is_valid_crawl_url(link):
                            if self.valves.DEBUG:
                                logger.debug(
                                    f"Skipping invalid URL in BFS: {link[:100]}"
                                )
                            continue
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
        max_urls: Optional[int] = None,
        __event_emitter__: Callable[[dict], Any] = None,
    ) -> dict:
        """Research‑filter crawl: starts from seed URLs, follows promising links."""
        if max_urls is None:
            max_urls = (
                self.user_valves.CRAWL4AI_MAX_URLS or self.valves.CRAWL4AI_MAX_URLS
            )

        max_pages = max_urls
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

        for source_url in start_urls[:5]:
            if results["total_pages"] >= max_pages:
                break

            # Validate source URL before crawling
            validated = await self._validate_url_pipeline(
                [source_url],
                query,
                check_keywords=False,
                __event_emitter__=__event_emitter__,
            )
            if not validated:
                continue

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
                skip_validation=True,  # already validated above
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

                    if max_tokens > 0 and total_tokens + page_tokens > max_tokens:
                        logger.warning(f"Token limit reached. Stopping research crawl.")
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

            scored_links = []
            for link in source_result.get("links", [])[:15]:
                # Filter invalid URLs before scoring
                if not self._is_valid_crawl_url(link):
                    if self.valves.DEBUG:
                        logger.debug(
                            f"Skipping invalid URL in research filter: {link[:100]}"
                        )
                    continue
                if results["total_pages"] >= max_pages:
                    break
                score = sum(1 for kw in keywords if kw in link.lower())
                if score > 0:
                    scored_links.append((link, score))

            scored_links.sort(key=lambda x: x[1], reverse=True)

            crawled = 0
            for link, _ in scored_links:
                if results["total_pages"] >= max_pages or crawled >= 3:
                    break
                if max_tokens > 0 and total_tokens >= max_tokens:
                    break
                if not include_external:
                    parsed_link = urlparse(link)
                    parsed_source = urlparse(source_url)
                    if (
                        parsed_link.netloc
                        and parsed_link.netloc != parsed_source.netloc
                    ):
                        continue

                # Validate link before crawling
                validated_link = await self._validate_url_pipeline(
                    [link],
                    query,
                    check_keywords=False,
                    __event_emitter__=__event_emitter__,
                )
                if not validated_link:
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
                    urls=[link],
                    query=query,
                    skip_validation=True,
                    __event_emitter__=__event_emitter__,
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

                        if max_tokens > 0 and total_tokens + page_tokens > max_tokens:
                            logger.warning(
                                f"Token limit reached. Stopping research crawl."
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
