#!/bin/sh

bash ./src/clean_repo.sh

python3 -m venv ~/sgoinfre/llm_venv
source ~/sgoinfre/llm_venv/bin/activate

if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo '✅ Virtual environment activated.'
    echo '⭐ Installing requirements...'
    echo "💻 Python version: $(which python)"
else
    echo '❌ Failed to activate virtual environment.'
fi

pip install -r requirements.txt
