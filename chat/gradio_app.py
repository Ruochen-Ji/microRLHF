import gradio as gr

def greet(name: str) -> str:
    name = name.strip() or "world"
    return f"Hello, {name}!"

with gr.Blocks(title="nanoGPT Hello") as demo:
    gr.Markdown("# nanoGPT Hello World")
    name_input = gr.Textbox(label="Name", placeholder="Type your name")
    output = gr.Textbox(label="Greeting", interactive=False)
    greet_btn = gr.Button("Say hello")
    greet_btn.click(greet, inputs=name_input, outputs=output)

if __name__ == "__main__":
    demo.launch()
