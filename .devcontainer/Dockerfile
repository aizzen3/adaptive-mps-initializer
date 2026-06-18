# Base image
FROM ghcr.io/astral-sh/uv:0.8.4-bookworm-slim

# Working directory inside Docker
WORKDIR /workspace

# Install git
RUN apt-get update && \
  apt-get install -y git

# Build tools for kahypar / scientific packages
RUN apt-get install -y build-essential libboost-all-dev cmake

# Clean up
RUN apt-get clean && \
  rm -rf /var/lib/apt/lists/*

# Open bash when container starts
CMD ["bash"]