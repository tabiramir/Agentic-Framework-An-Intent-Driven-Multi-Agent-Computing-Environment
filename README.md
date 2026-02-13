# Agentic-Framework-An-intent-driven-multi-agent-computing-environment

A Voice-Enabled Multi-Agent System
Transforming traditional computing into an intelligent, intent-driven operating environment.


# What is Agentic Framework?

It is a speech-enabled, multi-agent system prototype that allows users to interact with their computer using natural language. Instead of manually switching between apps, the system: 
Understands your intent
Breaks down complex goals
Coordinates specialized agents
Executes tasks autonomously
Built entirely in Python with a modular architecture.

# Key Features

● Voice-First Interaction - Wake word activation (“Hey Agent”), Real-time Voice Activity Detection (VAD), Whisper-based offline speech recognition, Text-to-speech responses

● Multi-Agent Architecture - Planner Agent (central coordinator), Reminder Agent, File Manager Agent, Web Search Agent, Mail Agent (Gmail integration), Booking Agent, Browser Control Agent, Process Monitor Agent, Sleep / Wake Control Agent, App Close Agent

● Intelligent NLP Pipeline - Intent refinement engine, nlu, Entity extraction (spaCy), Fuzzy matching (RapidFuzz), Date parsing (dateparser), Structured command normalization

● Real-Time GUI (NiceGUI) - Live transcription, Siri-style audio animation, Agent state visualization, Structured logs

# Architecture Overview

![PHOTO-2025-11-10-10-31-52](https://github.com/user-attachments/assets/cb3a45eb-2cb8-4846-9f74-1f72a26a21e7)

# Getting Started

Prerequisites 

● Python 3.0

● MongoDB

● OpenAI Whisper

# Setup 

    # Clone

    git clone https://github.com/tabiramir/Agentic-Framework-An-Intent-Driven-Multi-Agent-Computing-Environment.git

    cd Agentic-Framework-An-Intent-Driven-Multi-Agent-Computing-Environment

    # Create Virtual Environment

    python -m venv venv

    source venv/bin/activate      # Mac/Linux

    venv\Scripts\activate         # Windows

    # Install Dependencies

    pip install -r requirements.txt

    # Download spaCy model

    python -m spacy download en_core_web_sm

# Download OpenAi Whisper

    pip install git+https://github.com/openai/whisper.git 

It also requires the command-line tool ffmpeg to be installed on your system, which is available from most package managers:

    # on Ubuntu or Debian
    sudo apt update && sudo apt install ffmpeg

    # on Arch Linux
    sudo pacman -S ffmpeg

    # on MacOS using Homebrew (https://brew.sh/)
    brew install ffmpeg

    # on Windows using Chocolatey (https://chocolatey.org/)
    choco install ffmpeg

    # on Windows using Scoop (https://scoop.sh/)
    scoop install ffmpeg

# Running

     # GUI Mode

     python app_nicegui.py

     # Terminal

     python main.py

# Examples

● “Hey Agent, open Firefox”

● “Remind me to submit the report at 3 PM”

● “Search for multi-agent systems research”

● “Close Spotify”

● “Read my latest email”

# Why It’s Different

● Multi-agent orchestration

● Offline speech recognition

● Intent-driven execution

● Modular architecture

● Real-time GUI
    
# Future Improvements

● LLM-based reasoning layer

● Adaptive memory learning

● IoT device integration

● Reinforcement-learning planner




