#!/bin/bash

source /home/adminuser/venv/bin/activate
streamlit run --server.maxMessageSize 100 Deployment.py
