# Use a secure, long-term support Debian Bookworm base image with Python 3.12 pre-compiled
FROM mcr.microsoft.com/devcontainers/python:3.12-bookworm

# Avoid interactive installation prompts during container configuration build steps
ENV DEBIAN_FRONTEND=noninteractive

# Install critical low-level operating system network and testing binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
    netcat-openbsd \
    curl \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Pre-upgrade the core Python packet delivery application layers globally
RUN pip install --no-cache-dir --upgrade pip setuptools wheel