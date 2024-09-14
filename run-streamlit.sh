#!/bin/bash

sudo -E -u appuser /home/adminuser/venv/bin/streamlit --runtime-disable-gpu "$@"
