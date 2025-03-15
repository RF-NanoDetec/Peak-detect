#!/bin/bash

echo "Starting Peak Analysis Tool..."
python app.py

if [ $? -ne 0 ]; then
    echo "Error occurred during execution."
    echo "Check log files for details."
    read -p "Press Enter to continue..."
    exit 1
fi

exit 0 