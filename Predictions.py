import numpy as np
from keras.layers import TextVectorization
from keras.models import load_model
import gradio as gr
import pandas as pd
model = load_model("model.h5")
df=pd.read_csv('train.csv')
x=df['comment_text']
max_features = 200000  # number of words in the vocab dictionary
vectorizer = TextVectorization(max_tokens=max_features, output_sequence_length=1800, output_mode='int')
vectorizer.adapt(x.values)

def score_comment(comment):
    input_str=vectorizer(comment)
    results = model.predict(np.expand_dims(input_str,0))

    text = ''
    for idx, col in enumerate(['toxic', 'severe_toxic', 'obscene', 'thread', 'insult', 'identity_hate']):
        text += '{}: {}\n'.format(col, results[0][idx]>0.5)

    return text


interface = gr.Interface(fn=score_comment,
                         inputs=gr.inputs.Textbox(lines=2, placeholder='Enter your comment'),outputs='text')
interface.launch(share=True)