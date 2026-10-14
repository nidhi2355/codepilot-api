# Use the official Microsoft Python dev container image
FROM mcr.microsoft.com/devcontainers/python:1-3.11

# Ensure Python output is sent directly to the terminal without buffering
ENV PYTHONUNBUFFERED=1

# Remove the broken Yarn repository, then update and install dependencies
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
    postgresql-client