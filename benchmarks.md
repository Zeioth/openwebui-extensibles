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

But very likely there's other stuff we could optimize before recurring to that. Which an average of 200s per query, we need at least a x4 improvement to achieve an OK user experience.

Which is gonna be challenging on regular hardware.
