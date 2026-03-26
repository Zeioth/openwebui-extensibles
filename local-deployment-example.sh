# This is an example of running openwebui + search + crawler.

#!/bin/bash
# Re-Start ollama and open web ui.
# NOTA: Corre `ollama ps` para asegurarte del numero de modelos concurrentes.

## ----------- STOP
sudo docker stop ollama open-webui searxng crawl4ai
sudo docker rm ollama open-webui searxng crawl4ai

## ----------- START
sudo docker run -d \
  --device /dev/dri/card1 \
  --device /dev/dri/renderD128 \
  --device /dev/kfd \
  -e HSA_OVERRIDE_GFX_VERSION=12.0.1 \
  -e OLLAMA_KV_CACHE_TYPE=q4_0 \
  -e OLLAMA_FLASH_ATTENTION=1 \
  -e OLLAMA_NUM_PARALLEL=3 \
  -e OLLAMA_MAX_LOADED_MODELS=3 \
  -v ollama:/root/.ollama \
  -p 11434:11434 \
  --memory="20g" \
  --name ollama \
  --restart always \
  ollama/ollama:rocm

sudo docker run -d \
  -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --memory="20g" \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main

sudo docker run -d \
  -p 8888:8080 \
  -v "$HOME/.config/searxng/:/etc/searxng/" \
  -v "$HOME/.local/share/searxng/data/:/var/cache/searxng/" \
  --memory="30g" \
  --name searxng \
  --restart always \
	docker.io/searxng/searxng:latest

sudo docker run -d \
  -p 11235:11235 \
  --shm-size=2g \
  --memory="20g" \
  --add-host=host.docker.internal:172.17.0.1 \
  -v crawl4ai_data:/app/data \
  --name crawl4ai \
  --restart=always \
  unclecode/crawl4ai:latest

# Download models if needed.
nohup bash -c '
echo "Iniciando descargas..."
sleep 2 && sudo docker exec ollama env OLLAMA_NOHISTORY=1 ollama hf.co/second-state/gemma-3-12b-it-GGUF:gemma-3-12b-it-Q4_K_S.gguf
sleep 2 && sudo docker exec ollama env OLLAMA_NOHISTORY=1 ollama pull hf.co/John1604/Qwen3-14B-gguf:q4_k_s
sleep 2 && sudo docker exec ollama env OLLAMA_NOHISTORY=1 ollama pull huggingface.co/mradermacher/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_S
sleep 2 && sudo docker exec ollama env OLLAMA_NOHISTORY=1 ollama pull huggingface.co/mradermacher/Fast-Math-Qwen3-14B-GGUF:Q4_K_S
sleep 2 && sudo docker exec ollama env OLLAMA_NOHISTORY=1 ollama pull kwangsuklee/Qwen3.5-9B.Q4_K_M-Claude-4.6-Opus-Reasoning-Distilled-v2
sleep 2 && sudo docker exec ollama env OLLAMA_NOHISTORY=1 ollama pull Inference/Schematron:3B
sleep 2 && sudo docker exec ollama env OLLAMA_NOHISTORY=1 ollama pull huggingface.co/aman2024/NuExtract-2-2B-GGUF
echo "Todas las descargas completadas"
' > /tmp/ollama-models.log 2>&1 &

echo "Modelos descargando en segundo plano: /tmp/ollama-models.log"
