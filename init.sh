#!/bin/sh

activate_venv() {
    source ~/sgoinfre/llm_venv/bin/activate
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        echo '✅ Virtual environment activated.'
        echo "💻 Python location: $(which python) | Version: $(python --version 2>&1)"
        echo -e '⭐ Virtual environment ready\n'
    else
        echo '❌ Failed to activate virtual environment.'
        exit 1
    fi
}

setup_jupyter() {
    pip install jupyter notebook
    python -m ipykernel install --user --name=llm_venv
    echo "🔰 Jupyter configured with virtual environment kernel"
}

run_jupyter() {
    echo "🚀 Starting Jupyter Notebook..."
    jupyter notebook
}

full_init() {
    # Clean previous environment if exists
    if [ -d "~/sgoinfre/llm_venv" ]; then
        rm -rf ~/sgoinfre/llm_venv
    fi

    # Create new virtual environment
    python3 -m venv ~/sgoinfre/llm_venv
    activate_venv
    
    echo "📦 Installing requirements..."
    pip install -r requirements.txt
    
    setup_jupyter
}

case "$1" in
    -run)
        activate_venv
        run_jupyter
        ;;
    "")
        full_init
        ;;
    *)
        echo "❌ Invalid argument: $1"
        echo "Usage: source $0 [-run]"
        echo "  no args : Full initialization"
        echo "  -run    : Activate venv and run Jupyter"
        return 1
        ;;
esac