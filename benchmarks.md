Just for fun; don't take it super serious.

## Benchmarks v3.8.4

### Test 1 - `600 tokens` for crawl4AI and `2400 output tokens`. Model `NuExtract-2-2B-GGUF:Q3_K_M`
Given the prompt
> Search 'tipos de fresa' with research mode using the <x> strategy.

| research disabled | pseudo_adaptive | llm_guided | BFS-Deep| Research FILTER|
|-------------------|-----------------|------------|---------|-----------|
| 349s              |        495s     |    98s     |    72s  |     153s  |

All research modes have timed out due to excesive tokens usage. Let's repeat the test unlimitting all tokens.

### Test 2 - Unlimited tokens. Model hf.co/aman2024/NuExtract-2-2B-GGUF:Q3_K_M
Given the prompt
> Search 'tipos de fresa' with research mode using the <x> strategy.

| research disabled | pseudo_adaptive | llm_guided | BFS-Deep| Research FILTER|
|-------------------|-----------------|------------|---------|-----------|
| 538,9s               |        317.3s    |    171.347s     |  103.263s    |    251.1s |

We've noted 'research disabled' is sending batches instead of working in parallel like research mode. Because of this, it can overflow the system resources easily and cause timeouts and error 500 on crawl4ai. A temporal solution is either limit tokens to a small amount, or reduce batch size.
Also, if we keep batches, we should apply them to all modes. But I don't find a great reason for using this approach atm...

1. Maybe using batches so produce bigger context would take less tokens to find relevant context.

2. Or maybe... There's an even better strategy we could implement. Like two passes. One with a quick model for structured data and few tokens, to discover what sources are relevant, and then a second pass with a more capable model to extract the relevant information with better quality.

3. A third possiblity would be double down on step 1, and spend most tokens there to generate massive context (relevant, and irrelevant), and then a quick second step to produce a small, but very relevant response. There's risk to leave important info out this way though.

At the moment we are keeping thing simple and using simple model for scraping, and our real powerful model on the chat to process everything. So a second step might not even be necessary. But we should measure all these ideas. At least, mathematically, prior to manual testing, as each test take 2 minutes minimum.

I'm gonna save you the math, but two‑pass filtering is ~20% superior for the same token budget when r < 1. It allows concentrating tokens on the subset of pages that matter, yielding higher extraction quality per relevant page. It also uses fewer total tokens (if we keep the same budget, we can allocate even more tokens to the second pass, improving quality further).

But very likely there's other stuff we could optimize before recurring to that. Whith an average of 200s per query, we need at least a x4 improvement to achieve an OK user experience.

Overall our profiling level is quite good at base, and we are using 100% GPU. So the best way to improve performance would be reducing the amount of load. For that, prioritary things are:

- Reducing the amount of pages to scrape.
- For that we must ensure the quality of the ones we scrape.
- For that let's assume we can trust the ranking of the search engine.
- But for that to be true, we must be able to transport the chat's user request as literal as possible to the search engine. Otherwise, we can't make results predictible.

After that:
- Use aiohttp and a simple HTML parser (e.g., trafilatura or readability) to extract the first n characters from the body and find keywords from the user query.
- This is mostly useless, but it uses CPU time and we are using none so far, so we can likely leverage that on parallel, even. It should take 1-2s per page
- Then we send only those top *k* pages to _crawl_url for full GPU extraction. Because you are processing fewer pages, we can afford to give each one a larger token budget (increase CRAWL4AI_MAX_TOKENS per page or allocate more tokens overall).
- Extra: On research mode, we can use the top *k* pages as the seeds for the chosen research strategy. That way, deep crawling starts from the most promising sources.


# Why This Gives a 4× Speedup

    * Old: N = 10 pages, each heavy extraction ~20 s → 200 s total.
    * New: lightweight fetch for 10 pages takes ~2 s each in parallel → ~2–5 s total. Then heavy extraction on k = 2 pages → 40 s. Total ~45 s → 4.4× speedup.

Even if lightweight fetch takes 3–4 seconds per page (parallel), it’s still a small fraction compared to saving 160 s of GPU time.

If a page fails the lightweight fetch (low text), we can fallback to using Crawl4AI for that URL. This adds a bit of time but ensures we don’t miss important JS‑rendered content.

Example: after lightweight fetch, for pages with very little text (<200 chars) but a high initial ranking (e.g., among the top 3 search results), you could force‑include them for heavy extraction.

Vibe coded prototype:

Add this to the Tools class
```python
async def _lightweight_fetch(
    self, 
    url: str, 
    __event_emitter__: Callable = None
) -> dict:
    """
    Fetch a URL quickly, extract clean text, and return a simple relevance score.
    Returns a dict with keys: 'url', 'text', 'score', 'error'.
    """
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        headers = {
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
        }
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status != 200:
                    return {"url": url, "error": f"HTTP {resp.status}", "score": 0}
                html = await resp.text()
                # Use a fast text extractor (trafilatura, readability, or a simple regex)
                try:
                    import trafilatura
                    text = trafilatura.extract(html, include_comments=False, no_fallback=False)
                except ImportError:
                    # fallback: remove script/style tags, keep plain text
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    for tag in soup(["script", "style", "nav", "footer"]):
                        tag.decompose()
                    text = soup.get_text(separator=" ", strip=True)
                if not text or len(text) < 200:
                    return {"url": url, "error": "Too little text", "score": 0}
                # Compute simple relevance score (e.g., keyword frequency)
                keywords = self._query_keywords
                score = sum(text.lower().count(kw) for kw in keywords)
                return {"url": url, "text": text, "score": score}
    except Exception as e:
        return {"url": url, "error": str(e), "score": 0}
```

Then add this to the entry point search_and_crawl function 

```python
# Store query keywords for scoring
self._query_keywords = query.lower().split()

# Lightweight fetch all gathered URLs (parallel)
if __event_emitter__:
    await __event_emitter__({"type": "status", "data": {"description": "Quickly inspecting pages...", "done": False}})

light_tasks = [self._lightweight_fetch(url, __event_emitter__) for url in gathered_urls]
light_results = await asyncio.gather(*light_tasks)

# Sort by score and keep top k
k = 2  # can be a valve or user valve
scored = [r for r in light_results if r.get("score", 0) > 0]
scored.sort(key=lambda x: x["score"], reverse=True)
selected_urls = [r["url"] for r in scored[:k]]

if not selected_urls:
    # fallback to first few URLs (maybe the lightweight fetch failed for all)
    selected_urls = gathered_urls[:k]

if __event_emitter__:
    await __event_emitter__({"type": "status", "data": {"description": f"Selected {len(selected_urls)} most relevant pages.", "done": False}})
```

Such as:

```python
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
    :param urls: Optional list of specific URLs to crawl in addition to those found from searching.
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

    # Add user‑supplied URLs first
    if urls:
        for url in urls:
            if not url.startswith("http"):
                url = f"https://{url}"
            if url not in gathered_urls:
                gathered_urls.append(url)

    # Initial status update
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

    # ── Search for more URLs ─────────────────────────────────────────────────
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

    # Limit total URLs by user valve / max_urls
    max_urls = self.user_valves.CRAWL4AI_MAX_URLS or self.valves.CRAWL4AI_MAX_URLS
    if len(gathered_urls) > max_urls:
        gathered_urls = gathered_urls[:max_urls]

    # ── NEW: Lightweight pre‑filtering (CPU, parallel) ───────────────────────
    if __event_emitter__ and self.valves.MORE_STATUS:
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": "Quickly inspecting pages to find the most relevant...",
                    "done": False,
                },
            }
        )

    # Store query keywords for scoring
    self._query_keywords = query.lower().split()

    # Fetch each URL lightly and assign a score
    light_tasks = [self._lightweight_fetch(url, __event_emitter__) for url in gathered_urls]
    light_results = await asyncio.gather(*light_tasks)

    # Build a list of (url, score) for valid pages
    scored = [(res["url"], res["score"]) for res in light_results if res.get("score", 0) > 0]

    # Sort by descending score
    scored.sort(key=lambda x: x[1], reverse=True)

    # Determine how many pages to keep for deep crawling (default 2)
    k = 2  # You can make this configurable via a valve, e.g., self.valves.TOP_K_PAGES
    selected_urls = [url for url, _ in scored[:k]]

    # If we didn't get any good pages, fallback to the first few original URLs
    if not selected_urls:
        selected_urls = gathered_urls[:k]
        if self.valves.DEBUG:
            logger.warning(f"Lightweight filtering found no good pages, falling back to first {k} URLs")

    # Explicitly requested URLs should always be included, even if they scored low
    for url in urls or []:
        if url not in selected_urls:
            selected_urls.append(url)

    # Replace the original list with our filtered list
    gathered_urls = selected_urls

    if __event_emitter__ and self.valves.MORE_STATUS:
        await __event_emitter__(
            {
                "type": "status",
                "data": {
                    "description": f"Selected {len(gathered_urls)} most relevant pages for deep analysis.",
                    "done": False,
                },
            }
        )

    # ── Continue with the existing pipeline ─────────────────────────────────
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

    # Research mode
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

    # Standard batch crawl
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

                # Compile images
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

                # Compile videos
                for vid_url in crawled_batch.get("videos", []):
                    parsed_video = urlparse(vid_url)
                    base_video_url = f"{parsed_video.scheme}://{parsed_video.netloc}{parsed_video.path}"
                    if base_video_url in seen_videos:
                        continue
                    seen_videos.add(base_video_url)
                    video_list.append(vid_url)

                # Process content with token limits
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
                        if self.valves.DEBUG:
                            logger.info(
                                f"Truncated content from batch to {self.valves.CRAWL4AI_MAX_TOKENS} tokens"
                            )

                    if (
                        self.valves.CRAWL4AI_MAX_TOKENS > 0
                        and total_tokens + page_tokens > self.valves.CRAWL4AI_MAX_TOKENS
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

    # Final normalization
    crawl_results = self._normalize_content(crawl_results)

    if self.valves.DEBUG:
        logger.info(f"Final crawl_results count: {len(crawl_results)}")
        for idx, item in enumerate(crawl_results[:3]):
            logger.info(f"Sample {idx}: {type(item)} - {str(item)[:100]}")

    # Display media
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
```

(Solution not tested yet)
