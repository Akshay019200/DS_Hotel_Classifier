#!/bin/bash

source /home/adminuser/venv/bin/activate
export STREAMLIT_HEALTH_CHECK_TIMEOUT=300
streamlit run --verbose --server.maxMessageSize 100 Deployment.py
