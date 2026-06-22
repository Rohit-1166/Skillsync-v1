FROM python:3.11-slim

# HuggingFace Spaces requires a non-root user with ID 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy the requirements file and install dependencies
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files (HuggingFace automatically pulls Git LFS objects)
COPY --chown=user . .

# Expose port 7860 (Default for HuggingFace Spaces)
EXPOSE 7860

# Start the FastAPI server
CMD ["python", "-m", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "7860"]
