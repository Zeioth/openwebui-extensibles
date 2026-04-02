This fork fixes several bugs over the original project, adds better logging, and documentation. We normalize the crawled content to ensure no issue occur when processing data.

# 🌐 Web Search and Crawl for Open WebUI

**Version:** 2.8.4
**Author:** lexiismadd (with contributions from Zeioth)

This tool enables your Open WebUI instance to not only search the internet but to **deeply crawl, extract, and summarise** content from web pages. It combines the power of search engines (**SearXNG** and **Open WebUI Native Search**) with the advanced extraction capabilities of **Crawl4AI**.

---

## 📑 Table of Contents

- [✨ Key Features](#-key-features)
- [🚀 How It Works](#-how-it-works)
- [🤔 Examples](#-examples)
- [Configuration example](#configuration-example)
- [⚙️ Configuration Valves](#️-configuration-valves)
  - [Global Settings (Valves)](#global-settings-valves)
  - [Per‑User Settings (UserValves)](#peruser-settings-uservalves)
- [🛠️ Requirements](#️-requirements)
- [📖 Usage Examples](#-usage-examples)
- [🧩 Code Structure and Execution Flow](#-code-structure-and-execution-flow)
  - [Execution Flow](#execution-flow)
- [📝 Notes on Research Mode Activation](#-notes-on-research-mode-activation)
- [😉 Tips and Tricks](#-tips-and-tricks)
- [🐞 How to debug](#-how-to-debug)
- [⭐ Fork repo](#-fork-repo)

---

## ✨ Key Features

- **Dual‑Engine Search:** Simultaneously uses SearXNG and Open WebUI Native Search to find the most relevant URLs.
- **Intelligent Crawling:** Powered by [Crawl4AI](https://crawl4ai.com), it extracts clean, markdown‑formatted content while stripping away ads, sidebars, and navigation clutter.
- **Research Mode (Deep Crawling):** Recursively follows links on discovered pages to perform in‑depth research. Four strategies are available:
  - `pseudo_adaptive` – keyword‑based scoring and iterative crawling.
  - `llm_guided` – LLM‑assisted link selection (falls back to keyword scoring if LLM not used).
  - `bfs_deep` – breadth‑first deep crawling.
  - `research_filter` – relevance filtering.
- **LLM‑Driven Extraction:** Uses an OpenAI‑compatible LLM (like GPT‑4, Claude, or Ollama) to summarise and structure the crawled data before it reaches your chat.
- **Media Enrichment:** Automatically identifies and displays high‑quality images and videos from the sources, with clickable thumbnails directly in the chat.
- **Smart Token Management:** Automatic content truncation and token counting (via `tiktoken`) to keep responses within model limits – **now also applied in research mode**.
- **Concurrent Image Validation:** Validates image URLs in batches to ensure only “live” media is displayed.
- **Configuration Warnings:** Notifies you of common misconfigurations (missing protocols, malformed LLM provider) directly in the chat.
- **Highly Configurable:** Many valves to tune the tool to your preference and environment.
- **Organised Codebase:** Code is structured into logical regions for easier maintenance and extension.

---

## 🚀 How It Works

1. **Trigger:** When you ask a question requiring real‑time data, the LLM decides to call the tool.
2. **Gathering:** It queries SearXNG or Native Search and combines those results with any specific URLs you provided.
3. **Crawling:** It sends those URLs to a self‑hosted **Crawl4AI** instance.
4. **Extraction:** The configured LLM processes the raw HTML/Markdown into a structured summary following your extraction instructions.
5. **Delivery:** The tool returns a clean list of summaries, citations, and a gallery of relevant media to the chat interface.

---

# 🤔 Examples
The Crawl4AI LLM (you can specify in the valves) is smart enough to decide the best way to extract the content, but you can also do it manually if you want:

| Prompt example | Description |
|----------------------|-------------|
| `Search '<your_search>'` with research mode disabled | Research mode disabled |
| `Search '<your_search>'` with research mode | Keyword‑based scoring (pseudo_adaptive, default) |
| `Search '<your_search>'` with research mode using the pseudo_adaptive strategy | Keyword‑based scoring |
| `Search '<your_search>'` with research mode using the llm_guided strategy | LLM selects links |
| `Search '<your_search>'` with research mode using the bfs_deep strategy | Breadth‑first exploration |
| `Search '<your_search>'` with research mode using the research_filter strategy | URL filtering |


## Configuration example
In the function valves (assuming you use docker). 

| Valve | Value | Comments |
|--|--|--|
| SearXNG Search URL | http://host.docker.internal:8888/search?format=json&q=<query> | Assuming local searxng dockerized |
| Crawl4AI Base URL | http://host.docker.internal:11235 | Assuming local crawl4ai dockerized |
| LLM Base URL |  http://host.docker.internal:11434 | Assuming local ollama dockerized |
| LLM Provider and model  | ollama/hf.co/aman2024/NuExtract-2-2B-GGUF:Q3_K_M | Assuming ollama. The format is `backend/model`, so, ollama. |
| LLM Temperature | 0 | Recommended value for NuExtract or Schematron models. |
| Max Tokens used by Crawl4AI  | 1200 | Recommended to specify it so Crawl4AI doesn't take forever. This controls the reasoning during crawl4AI. |
| LLM Max Tokens  | 1200 | Recommended to specify it so Crawl4AI doesn't take forever. This controls the answer lenght. |
| Debug | true | Recommended unless you are going for production (which you should't). |

## ⚙️ Configuration Valves

### Global Settings (Valves)

| Category | Key | Description |
|----------|-----|-------------|
| **General** | `INITIAL_RESPONSE` | Message shown when the tool starts. Set empty to disable. |
| **Search** | `USE_NATIVE_SEARCH` | Enable/Disable Open WebUI's internal search. |
| | `SEARCH_WITH_SEARXNG` | Enable/Disable SearXNG integration. |
| | `SEARXNG_BASE_URL` | Full URL for your SearXNG API. Include `http://` or `https://` and replace the query with `<query>`. **Examples:**<br> `http://searxng:8888/search?format=json&q=<query>`<br> `http://host.docker.internal:8888/search?format=json&q=<query>`<br> **⚠️ Required when SearXNG is enabled.** |
| | `SEARXNG_API_TOKEN` | API token for your SearXNG instance. |
| | `SEARXNG_METHOD` | HTTP method for SearXNG calls (GET or POST). |
| | `SEARXNG_TIMEOUT` | Timeout (seconds) for SearXNG requests. |
| | `SEARXNG_MAX_RESULTS` | Maximum number of results from SearXNG. |
| **Crawl4AI** | `CRAWL4AI_BASE_URL` | URL of your Crawl4AI instance. Include `http://` or `https://`. **Examples:**<br> `http://crawl4ai:11235`<br> `http://host.docker.internal:11235` |
| | `CRAWL4AI_USER_AGENT` | Custom User‑Agent for Crawl4AI. |
| | `CRAWL4AI_TIMEOUT` | Timeout (seconds) for Crawl4AI requests. |
| | `CRAWL4AI_BATCH` | Number of URLs to send per batch. |
| | `CRAWL4AI_MAX_URLS` | Maximum URLs to crawl. |
| | `CRAWL4AI_EXTERNAL_DOMAINS` | Allow crawling external domains. |
| | `CRAWL4AI_EXCLUDE_DOMAINS` | Comma‑separated list of domains to exclude. |
| | `CRAWL4AI_EXCLUDE_SOCIAL_MEDIA_DOMAINS` | Social media domains to exclude. |
| | `CRAWL4AI_EXCLUDE_IMAGES` | Exclude images (None, External, All). |
| | `CRAWL4AI_WORD_COUNT_THRESHOLD` | Minimum word count for content to be included. |
| | `CRAWL4AI_TEXT_ONLY` | Extract only text, exclude images and media. |
| | `CRAWL4AI_DISPLAY_MEDIA` | Show images and videos in chat. |
| | `CRAWL4AI_MAX_MEDIA_ITEMS` | Maximum number of media items to show (0 = unlimited). |
| | `CRAWL4AI_DISPLAY_THUMBNAILS` | Show thumbnails instead of full images. |
| | `CRAWL4AI_THUMBNAIL_SIZE` | Thumbnail size in pixels (square). |
| | `CRAWL4AI_MIN_IMAGE_SCORE` | Minimum image score (0‑10) to include. |
| | `CRAWL4AI_VALIDATE_IMAGES` | Check if image URLs are accessible. |
| | `CRAWL4AI_MAX_TOKENS` | Maximum tokens for the entire web content response (0 = unlimited). |
| **LLM** | `LLM_BASE_URL` | Base URL for your OpenAI‑compatible LLM. Include `http://` or `https://`. **Examples:**<br> OpenRouter: `https://openrouter.ai/api/v1`<br> OpenAI: `https://api.openai.com/v1`<br> Ollama (Docker): `http://host.docker.internal:11434` |
| | `LLM_API_TOKEN` | API token for the LLM. |
| | `LLM_PROVIDER` | Provider/model in format: `<provider>/<model>`. **Examples:**<br> `openai/gpt-4o`<br> `ollama/llama3.2`<br> `openrouter/@preset/default`<br> **⚠️ For Ollama, you must include `ollama/` prefix.** |
| | `LLM_TEMPERATURE` | LLM temperature. |
| | `LLM_INSTRUCTION` | Instruction for LLM extraction. |
| | `LLM_MAX_TOKENS` | Maximum tokens for LLM output. |
| | `LLM_TOP_P` | Top P sampling. |
| | `LLM_FREQUENCY_PENALTY` | Frequency penalty. |
| | `LLM_PRESENCE_PENALTY` | Presence penalty. |
| **Other** | `MORE_STATUS` | Show extra status messages. |
| | `DEBUG` | Enable debug logging. |

### Per‑User Settings (UserValves)

These settings can be overridden per user (if the tool is configured to allow it). They appear in the user’s settings panel.

| Key | Description |
|-----|-------------|
| `SEARXNG_MAX_RESULTS` | Overrides the global `SearXNG Max Results`. |
| `CRAWL4AI_MAX_URLS` | Overrides the global `Crawl4AI Maximum URLs to crawl`. |
| `CRAWL4AI_DISPLAY_MEDIA` | Overrides the global `Display Media in Chat`. |
| `CRAWL4AI_MAX_MEDIA_ITEMS` | Overrides the global `Max Media Items to Display`. |
| `CRAWL4AI_DISPLAY_THUMBNAILS` | Overrides the global `Display images as thumbnails`. |
| `CRAWL4AI_THUMBNAIL_SIZE` | Overrides the global `Image thumbnail size`. |
| `RESEARCH_MODE` | Enable research mode (deeper crawling). |
| `RESEARCH_CRAWL_MODE` | Strategy: `pseudo_adaptive`, `llm_guided`, `bfs_deep`, `research_filter`. |
| `RESEARCH_KEYWORD_WEIGHT` | Weight for keyword relevance scoring. |
| `RESEARCH_MAX_DEPTH` | Maximum depth of links to follow (1–10). |
| `RESEARCH_MAX_PAGES` | Maximum pages to crawl (1–25). |
| `RESEARCH_BATCH_SIZE` | URLs per batch during research crawling. |
| `RESEARCH_LLM_LINK_SELECTION` | Use LLM to select next links (for `llm_guided`). |
| `RESEARCH_INCLUDE_EXTERNAL` | Allow following external domains during research. |

---

## 🛠️ Requirements

- **Crawl4AI Server:** A running instance of Crawl4AI (usually via Docker).  
  Default: `http://crawl4ai:11235`  
  **⚠️ Include `http://` or `https://` in the URL.**
- **SearXNG:** An accessible instance of SearXNG (if using SearXNG).  
  Default: `http://searxng:8888/search?format=json&q=<query>`  
  **⚠️ Include `http://` or `https://` and replace the query with `<query>`.**
- **Open WebUI:** A recent version (v0.6.42 or higher) to support Native Search and Tools.

---

## 📖 Usage Examples

**Standard Search:**

> “Find the latest news on SpaceX Starship and summarise the key findings.”

**Targeted Crawl:**

> “Crawl [https://example.com](https://example.com) and tell me their pricing structure.”

**Research Mode (activated by LLM or user):**

> “Perform a deep research search on ‘Ambient Computing’ and find at least 10 sources.”

---

## 🧩 Code Structure and Execution Flow

The code is organised into visual regions for easier maintenance:

1. **IMPORTS** – All required libraries.
2. **AUXILIARY CLASSES** – `ArticleData` (Pydantic model for extraction) and `ResearchCrawlMode` (constants).
3. **MAIN CLASS Tools** – Contains all logic:
   - **Valves / UserValves** – Configuration models with validation.
   - **Initialization and Auto‑configuration** – Normalises URLs, detects Docker, warns about misconfigurations.
   - **Content Helpers** – `_normalize_content` ensures consistent output; `_count_tokens` and `_truncate_content` manage token limits.
   - **Image Validation** – `_validate_image_url` and `_validate_images_batch` ensure media is accessible.
   - **Search** – `_search_native` and `_search_searxng` gather URLs.
   - **Main Entry Point** – `search_and_crawl` orchestrates search, crawling, token management, and media display.
   - **Research Mode** – `_research_crawl` dispatches to the selected strategy, all of which now respect the global token limit.
   - **Research Strategies** – Four methods implementing different crawling approaches, each with token‑aware content processing.
   - **Core Crawling** – `_crawl_url` sends URLs to Crawl4AI and returns content.

### Execution Flow

- The LLM calls `search_and_crawl` with a query and optional parameters.
- The tool validates the configuration and notifies the user of any warnings.
- URLs are gathered from user input, native search, and SearXNG.
- If research mode is enabled (by the LLM or user), a deep‑crawl strategy is used; otherwise, URLs are processed in batches.
- For each batch, `_crawl_url` sends a request to Crawl4AI, which returns extracted content, images, and videos.
- Content is normalised, token limits are enforced (truncation and early stopping), and media is validated.
- Finally, the results are returned to the LLM.

---

## 📝 Notes on Research Mode Activation

Research mode can be activated in two ways:
- **By the LLM:** When the LLM calls the tool, it can set the `research_mode` parameter to `true` (and optionally choose a `research_crawl_mode`).
- **By the user:** If a user enables `RESEARCH_MODE` in their `UserValves`, the tool will always run in research mode regardless of the LLM’s choice.

If both are set, the LLM’s `research_mode` takes precedence. 

These settings are not currently exposed to the user trough valves. The LLM will use them automatically, or though the prompt if the user want to enforce it.

---

## 😉 Tips and tricks
In order to speed up crawling, it's recommended to set your `LLM_PROVIDER` to something specialized for this kind of task. Good options are: 

`hf.co/aman2024/NuExtract-2-2B-GGUF:Q3_K_M` (Which supports images and its faster) or `Inference/Schematron:3B` (Text only but more powerful).

## 🐞 How to debug
If you find an issue configuring or using this function; Enable the debug valve and run this to see the logs

> sudo docker logs -f open-webui


Assuming you use docker.

## ⭐ Fork repo

Please log any issues [on the GitHub repository](https://github.com/Zeioth/openwebui-extensibles).

## TODOS

* Often, out of 10 search results, there is 1-2 of them highly superior in content. Therefore it's almost certain doing a second llm pass of those 1-2 best sites using extra tokens will increase the relevance of response considerably.
