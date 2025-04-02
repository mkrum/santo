# Santo justfile

# Build the Docker container
build:
    docker build -t santo .

# Run all tests
test:
    python -m pytest tests/

# Run the container
run:
    docker run santo