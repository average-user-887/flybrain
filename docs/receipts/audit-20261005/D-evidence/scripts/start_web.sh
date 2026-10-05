#!/bin/bash
exec <home>/Documents/ChatGPT/flybrain/.venv/bin/python -u -m http.server 8840 --bind 127.0.0.1 --directory "<repo>/web" > <scratch>/web.log 2>&1
