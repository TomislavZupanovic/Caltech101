from source.model import Model
from source.utils import random_image
import gradio as gr
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--usage', type=str, default='random',
                    choices=['random', 'gradio'],
                    help='Usage mode')
args = parser.parse_args()
model = Model()  # Instantiate model
model.load_weights('model_Caltech101')

if args.usage == 'gradio':
    def gr_predict(inp):
        classes, top_prob = model.predict(inp)
        return {classes[i]: float(top_prob[i]) for i in range(5)}
    inputs = gr.inputs.Image(type='pil')
    outputs = gr.outputs.Label(num_top_classes=4)
    gr.Interface(fn=gr_predict, inputs=inputs, outputs=outputs).launch()
else:
    model.predict(random_image(), plot_pred=True)



