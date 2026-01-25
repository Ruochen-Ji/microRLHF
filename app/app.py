"""
MicroRLHF App Entry Point

Main Gradio/Streamlit application that combines all tabs:
- Inference Tab: Chat interface with tokens/sec display
- Training Tab: Streaming loss curves, hyperparameter controls
- Finetune Tab: SFT and LoRA options
- Annotate Tab: Preference data collection UI

Usage:
    python app/app.py
"""

# TODO: Phase 1 Implementation
# - Set up Gradio app with tab navigation
# - Import and integrate all tab modules
# - Add checkpoint save/load functionality


def create_app():
    """Create and configure the main application."""
    raise NotImplementedError("Phase 1: Set up Gradio/Streamlit app")


def main():
    """Entry point for the application."""
    app = create_app()
    app.launch()


if __name__ == "__main__":
    main()
