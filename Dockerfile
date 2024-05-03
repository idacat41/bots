# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/app/torch
RUN mkdir -p ${TORCH_HOME}/hub
RUN chmod -R 777 /app/
# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    vim \
    ffmpeg \
    libffi-dev \
    python3-dev \
    git \
    python3-venv \
    build-essential
    

# Set the working directory
WORKDIR /app

# Copy the application files to the container
COPY . /app
RUN bash ./install_voice_recv.sh
RUN pip3 install -r requirements.txt
# Clone the discord.py repository and install it
RUN python -m pip install git+https://github.com/Rapptz/discord.py

 # Creates a non-root user with an explicit UID and adds permission to access the /app folder
# # For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app

# Create the /app/config directory and set ownership to appuser
RUN mkdir -p /app/config && chown -R 5678:5678 /app/config

# Switch to the appuser
USER appuser

# During debugging, this entry point will be overridden.
# For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["sh", "-c", "clear; python app.py | logger -t app.log"]
