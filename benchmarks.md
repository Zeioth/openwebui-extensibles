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

```
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

Then modify the crawler function

```
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
