# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install vim for editing files if necessary
RUN apt-get update && apt-get install -y vim ffmpeg libffi-dev python3-dev git python3-venv
ARG VIRTUAL_ENV=/opt/venv
RUN pip3 install uv
RUN uv venv ${VIRTUAL_ENV}
ENV PATH="${VIRTUAL_ENV}/bin:$PATH"
ENV PYTHON_ENV=${VIRTUAL_ENV}
COPY install_discord.sh .
# Install pip requirements
RUN chmod +x ./install_discord.sh
RUN ./install_discord.sh
COPY requirements.txt .
RUN uv pip install -r requirements.txt

WORKDIR /app
COPY . /app

# # Creates a non-root user with an explicit UID and adds permission to access the /app folder
# # For more info, please refer to https://aka.ms/vscode-docker-python-configure-containers
RUN adduser -u 5678 --disabled-password --gecos "" appuser && chown -R appuser /app

# Create the /app/config directory and set ownership to appuser
RUN mkdir -p /app/config && chown -R 5678:5678 /app/config

# Switch to the appuser
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["python", "app.py", "|", "logger", "-t", "app.py"]
