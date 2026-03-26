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
| 538,9s               |        317.3 s    |    171.347 s     |  103.263 s    |    251.1 s |

We've noted 'research disabled' is sending batches instead of working in parallel like research mode. Because of this, it can overflow the system resources easily and cause timeouts and error 500 on crawl4ai. A temporal solution is either limit tokens to a small amount, or reduce batch size.
Also, if we keep bathces, we should apply them to all modes. But I don't find a great reason for using this approach atm...
