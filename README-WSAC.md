# 🌐 Web Search and Crawl for Open WebUI

**Version:** 2.8.3

**Author:** lexiismadd (with improvements by Zeioth)

This tool enables your Open WebUI instance to not only search the internet but to **deeply crawl, extract, and summarise** content from web pages. It combines the power of search engines (**SearXNG** and *Open WebUI Native Search*) with the advanced extraction capabilities of **Crawl4AI**.

---

## ✨ Key Features

- **Dual-Engine Search:** Simultaneously utilises [SearXNG](https://github.com/searxng/searxng) and OpenWebUI Native Search to find the most relevant URLs.
- **Intelligent Crawling:** Powered by [Crawl4AI](https://crawl4ai.com), it extracts clean, markdown-formatted content while stripping away ads, sidebars, and navigation clutter.
- **Research Mode:** Recursively follows links on discovered pages to perform "Deep Research". Four strategies are available:
  - `pseudo_adaptive` – keyword‑based scoring and iterative crawling.
  - `llm_guided` – LLM‑assisted link selection (placeholder, falls back to keyword scoring).
  - `bfs_deep` – breadth‑first deep crawling.
  - `research_filter` – relevance filtering.
- **LLM-Driven Extraction:** Uses an OpenAI-compatible LLM (like GPT-4, Claude, or Ollama) to summarise and structure the crawled data before it reaches your chat.
- **Media Enrichment:** Automatically identifies and displays high‑quality images and videos from the sources, providing clickable thumbnails directly in the chat. (*User‑specific setting*)
- **Smart Token Management:** Automatic content truncation and token counting (via `tiktoken`) to keep responses within model limits.
- **Concurrent Validation:** Validates image URLs in batches to ensure only “live” media is displayed.
- **Configuration Warnings:** Notifies the user of common misconfigurations (missing protocols, malformed LLM provider) directly in the chat.
- **Highly Configurable:** Many configuration valves to tune the tool to your preference and environment.

---

## 🚀 How It Works

1. **Trigger:** When you ask a question requiring real‑time data, the tool is called.
2. **Gathering:** It queries SearXNG or Native Search and combines those results with any specific URLs you provided.
3. **Crawling:** It sends those URLs to a self‑hosted **Crawl4AI** instance.
4. **Extraction:** The LLM configured in the “Valves” processes the raw HTML/Markdown into a structured summary focusing on your core instructions.
5. **Delivery:** The tool returns a clean list of summaries, citations, and a gallery of relevant media to the chat interface.

---

## ⚙️ Configuration Valves

### Global Settings (Valves)

| Category | Key | Description |
|----------|-----|-------------|
| **General** | `Initial delta response` | The message shown in chat when the tool starts working. Set empty to disable. |
| **Search** | `Use Native Search` | Enable/Disable OpenWebUI's internal search. |
| | `Search with SearXNG` | Enable/Disable SearXNG integration. |
| | `SearXNG Search URL` | Full URL for your SearXNG API. Include `http://` or `https://` and replace the query with `<query>`. **Examples**:<br> `http://searxng:8888/search?format=json&q=<query>`<br> `http://host.docker.internal:8888/search?format=json&q=<query>`<br> **⚠️ Required when SearXNG is enabled.** |
| | `SearXNG API Token` | The API token or Secret for your SearXNG instance. |
| | `SearXNG HTTP Method` | HTTP method to use for SearXNG API calls (GET or POST). |
| | `SearXNG Timeout` | The timeout (in seconds) for SearXNG API requests. |
| | `SearXNG Max Results` | Maximum number of results to return from SearXNG. |
| **Crawl4AI** | `Crawl4AI Base URL` | The URL of your Crawl4AI Docker/Server instance. Include `http://` or `https://`. **Examples**:<br> `http://crawl4ai:11235`<br> `http://host.docker.internal:11235` |
| | `Crawl4AI User Agent` | Custom User-Agent string for Crawl4AI. |
| | `Crawl4AI Timeout` | The timeout (in seconds) for Crawl4AI requests. |
| | `Crawl4AI Batch` | The number of URLs to send to Crawl4AI per batch. |
| | `Crawl4AI Maximum URLs to crawl` | The maximum number of URLs to crawl with Crawl4AI. |
| | `Crawl External Domains` | Allow Crawl4AI to crawl external/additional URL domains. |
| | `Excluded Domains` | Comma-separated list of external domains to exclude from crawling. |
| | `Excluded Social Media Domains` | Comma-separated list of social media domains to exclude from crawling. |
| | `Exclude Images` | Exclude images from crawling (None, External, All). |
| | `Word Count Threshold` | The word count threshold for content to be included. |
| | `Text Only` | Only extract text content, excluding images and other media. Disables crawling and displaying media in the chat. |
| | `Display Media in Chat` | Display images and videos as clickable links in the chat window. |
| | `Max Media Items to Display` | Maximum number of images/videos to display (0 = unlimited). |
| | `Display images as thumbnails` | Display images as thumbnails in the chat window. |
| | `Image thumbnail size` | Image thumbnail size (in px) square. Ignored if thumbnails are off. |
| | `Min Image Score To Include` | Minimum image score (0–10) from Crawl4AI to include an image. |
| | `Validate Image Links` | Verify image links before displaying. |
| | `Max Tokens used by web content` | Maximum tokens for the entire web content response (0 = unlimited). |
| **LLM** | `LLM Base URL` | The base URL for your preferred OpenAI-compatible LLM. Include `http://` or `https://`. **Examples**:<br> OpenRouter: `https://openrouter.ai/api/v1`<br> OpenAI: `https://api.openai.com/v1`<br> Ollama (Docker): `http://host.docker.internal:11434` |
| | `LLM API Token` | Optional API Token for your preferred OpenAI-compatible LLM. |
| | `LLM Provider` | The LLM provider and model to use. Format: `<provider>/<model-name>`. **Examples**:<br> `openai/gpt-4o`<br> `ollama/llama3.2`<br> `openrouter/@preset/default`<br> **⚠️ For Ollama, you must include `ollama/` prefix.** The tool may auto‑correct if you forget. |
| | `LLM Temperature` | The temperature to use for the LLM. |
| | `LLM Extraction Instruction` | The instruction to use for the LLM when extracting from the webpage. |
| | `LLM Max Tokens` | The maximum number of tokens to use for the LLM. |
| | `LLM Top P` | The top_p value to use for the LLM. |
| | `LLM Frequency Penalty` | The frequency penalty to use for the LLM. |
| | `LLM Presence Penalty` | The presence penalty to use for the LLM. |
| **Other** | `More status updates` | Show more status updates during web search and crawl. |
| | `Debug logging` | Enable detailed debug logging. |

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

To use this tool, your environment must have:

- **Crawl4AI Server:** A running instance of Crawl4AI (usually via Docker).  
  Default: `http://crawl4ai:11235`  
  **⚠️ Include `http://` or `https://` in the URL.**
- **SearXNG:** An accessible instance of SearXNG (if using SearXNG).  
  Default: `http://searxng:8888/search?format=json&q=<query>`  
  **⚠️ Include `http://` or `https://` and replace the query with `<query>`.**
- **OpenWebUI:** A recent version (v0.6.42 or higher) to support Native Search and Tools.

---

## 📖 Usage Example

**Standard Search:**

> “Find the latest news on SpaceX Starship and summarise the key findings.”

**Targeted Crawl:**

> “Crawl [https://example.com](https://example.com) and tell me their pricing structure.”

**Research Mode:**

> “Perform a deep research search on ‘Ambient Computing’ and find at least 10 sources.”

---

## 🧩 Code Structure and Execution Flow

The code is organised into visual regions for easier maintenance:

1. **IMPORTS** – All required libraries.
2. **AUXILIARY CLASSES** – `ArticleData` (Pydantic model for extraction) and `ResearchCrawlMode` (constants).
3. **MAIN CLASS Tools** – Contains all logic:
   - **Valves / UserValves** – Configuration models with validation.
   - **Initialization and Auto‑configuration** – Normalises URLs, detects Docker, auto‑configures Ollama if needed.
   - **Helpers for Notifications and Logging** – `_emit_*` and `_log_and_emit_*` methods centralise frontend updates and logging.
   - **URL and Content Normalization** – `_normalize_content` ensures consistent output format.
   - **Token Counting and Truncation** – `_count_tokens`, `_truncate_content`.
   - **Image Validation** – `_validate_image_url`, `_validate_images_batch`.
   - **Native Search** – `_search_native` calls OpenWebUI’s built‑in search.
   - **SearXNG Search** – `_search_searxng` performs GET/POST to SearXNG.
   - **Main Entry Point** – `search_and_crawl` orchestrates search, crawling, token management, and media display.
   - **Research Mode** – `_research_crawl` dispatches to the selected strategy.
   - **Research Strategies** – Four methods implementing the different crawling approaches.
   - **Core Crawling** – `_crawl_url` sends URLs to Crawl4AI and returns content.
   - **Configuration Notification** – `_notify_configuration_issues` checks for common misconfigurations and warns the user.

### Execution Flow

- The LLM calls `search_and_crawl` with a query and optional parameters.
- The tool validates the configuration and notifies the user of any warnings.
- URLs are gathered from user input, native search, and SearXNG.
- If research mode is enabled, a deep‑crawl strategy is used; otherwise, URLs are processed in batches.
- For each batch, `_crawl_url` sends a request to Crawl4AI, which returns extracted content, images, and videos.
- Content is normalised, token limits are enforced, and media is validated.
- Finally, the results are returned to the LLM.

---

## Issues

Please log any issues [on my Github repo](https://github.com/lexiismadd/openwebui-extensibles/issues)
