#!/bin/bash

# Install Python dependencies
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Failed to install requirements."
    exit 1
fi

# Run the first script
python3 create_mitre_techniques.py
if [ $? -ne 0 ]; then
    echo "create_mitre_techniques.py failed."
    exit 1
fi

# Run the second script
python3 generate_commands.py
if [ $? -ne 0 ]; then
    echo "generate_commands.py failed."
    exit 1
fi

echo "All tasks completed successfully!"
