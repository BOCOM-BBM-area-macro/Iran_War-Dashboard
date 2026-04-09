#!/bin/bash
# Convenience script to run PeekDeck pipeline

export PYTHONPATH="${PYTHONPATH}:${PWD}/src"

# If running 'all', delete entire data folder for clean testing
if [ "$1" = "all" ]; then
    echo "🗑️  Deleting data folder for clean rebuild..."
    rm -rf data
fi

python -m peek_deck "$@"
