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
| 744.3 s              |        317.3 s    |         |      |    251.1 s |
