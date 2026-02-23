# Import the pipeline utility from Hugging Face Transformers
# 'pipeline' makes it easy to use pre-trained models for common NLP tasks
from transformers import pipeline

# Import Gradio for building the web-based user interface
import gradio as gr

# Load a pre-trained summarization model
# By default this uses 'sshleifer/distilbart-cnn-12-6', a lightweight summarization model
model = pipeline("summarization")

# Define the prediction function that will be called when the user submits text
def predict(prompt):
    # Run the input text through the model
    # The result is a list of dicts; [0] gets the first result, ["summary_text"] extracts the summary
    summary = model(prompt)[0]["summary_text"]
    return summary

# Create a Gradio Blocks layout to define the UI
with gr.Blocks() as demo:
    # Create a multi-line text input box for the user to paste text into
    textbox = gr.Textbox(placeholder="Enter text block to summarize", lines=4)

    # Wire up the interface: takes the textbox as input, runs predict(), and displays text output
    gr.Interface(fn=predict, inputs=textbox, outputs="text")

# Launch the app on a local web server (opens in the browser automatically)
demo.launch()
