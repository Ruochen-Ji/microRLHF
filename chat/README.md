# nanoGPT Chat Interface

A Gradio-based web interface for interacting with nanoGPT.

## Starting the Server

From the project root directory, run:

```bash
python chat/gradio_app.py
```

Or navigate to the chat directory first:

```bash
cd chat
python gradio_app.py
```

The server will start and display a local URL (typically `http://127.0.0.1:7860`). Open this URL in your browser to access the interface.

## Options

You can customize the server launch with additional arguments:

```bash
# Share publicly (creates a temporary public URL)
python chat/gradio_app.py --share

# Specify a custom port
python chat/gradio_app.py --server-port 8080

# Allow external connections
python chat/gradio_app.py --server-name 0.0.0.0
```
