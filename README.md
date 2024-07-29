Ollama:
* Linux: `curl -fsSL https://ollama.com/install.sh | sh`
* Mac: https://ollama.com/download/mac

Install:
* `pip install -r requirements.txt`
* `ollama pull llama3.1`

Make IPA data:
* `rm ./data/lyrics_ipa.txt && ./run.py translate`
* `./run.py clean`

Train:
* `./run.py train`

Generate:
* `./run.py gen`


