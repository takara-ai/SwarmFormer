from swarmformer.inference import load_trained_model, inference
from swarmformer.config import MODEL_CONFIGS

from transformers import AutoTokenizer
import gradio as gr

HEADER_MARKDOWN = open('header.md', 'rt').read()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

base_model = load_trained_model(MODEL_CONFIGS['base'], 'cpu')
small_model = load_trained_model(MODEL_CONFIGS['small'], 'cpu')

def classify(text: str):
    base_model_result = inference(base_model, tokenizer, text)[2]
    small_model_result = inference(small_model, tokenizer, text)[2]
    return [base_model_result, small_model_result]

with gr.Blocks() as demo:
    gr.Markdown(value=HEADER_MARKDOWN)
    with gr.Row('compact'):
        input_string = gr.Textbox(placeholder='The movie was awesome!', max_lines=1)
        with gr.Column():
            run_button = gr.Button("Run", variant='primary')
            clear_button = gr.ClearButton()
            
    with gr.Tab(label='Inference Results'):
        with gr.Row():
            json_output_base = gr.JSON(label='JSON Output (BASE)')
            json_output_small = gr.JSON(label='JSON Output (SMALL)')
    
    
    run_button.click(fn=classify, inputs=[input_string], outputs=[json_output_base, json_output_small])
demo.launch()