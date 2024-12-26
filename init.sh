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

run_model() {
    MODEL_PID=$(pgrep -f "python.*sentiment_analysis.py" || echo "")

    if [ ! -z "$MODEL_PID" ]; then
        echo "🔄 Stopping existing model process (PID: $MODEL_PID)..."
        kill -9 $MODEL_PID
    fi

    echo "🚀 Starting sentiment analysis model..."
    export PYTHONPATH=$(pwd):$PYTHONPATH
    
    cd src
    python sentiment_analysis.py
    cd ..
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

    run_model
}

case "$1" in
    -run)
        activate_venv
        run_model
        ;;
    "")
        full_init
        ;;
    *)
        echo "❌ Invalid argument: $1"
        echo "Usage: source $0 [-run]"
        echo "  no args : Full initialization"
        echo "  -run    : Just activate venv and run model"
        return 1  
        ;;
esac