# Install Ollama (Linux/Mac)
`curl -fsSL https://ollama.com/install.sh | sh`

# Pull a Bengali-capable model (pick one)
`ollama pull qwen2.5:14b      # better quality, needs more RAM/VRAM`

`ollama pull qwen2.5:7b       # lighter, still decent on Bengali`

`ollama pull gemma2:9b        # alternative option`

# Start the server (usually auto-starts, but just in case)
`ollama serve`

Test it's alive: `curl http://localhost:11434/api/tags` should list your pulled models.

# Run it

`pip install requests`

# Test on a few samples first
`python generate_qa.py --input "dataset samples.json" --output questions.md --model qwen2.5:14b --limit 5`

# Once happy, run on the full file
`python generate_qa.py --input "dataset samples.json" --output questions.md --model qwen2.5:14b`