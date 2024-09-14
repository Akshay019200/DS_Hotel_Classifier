#!/bin/bash

source /home/adminuser/venv/bin/activate
export STREAMLIT_HEALTH_CHECK_TIMEOUT=300

# Add logging and error handling
exec > >(tee /var/log/streamlit.log) 2>&1
set -e

streamlit run --verbose --server.maxMessageSize 100 Deployment.py
