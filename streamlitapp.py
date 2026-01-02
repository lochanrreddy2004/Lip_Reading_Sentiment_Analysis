# Import all of the dependencies
import streamlit as st
import os 
import imageio 
import numpy as np

import tensorflow as tf 
from utils import load_data, num_to_char
from modelutil import load_model
from sentimentutil import analyze_sentiment


# Set the layout to the streamlit app as wide 
st.set_page_config(layout='wide')

# # Setup the sidebar
# with st.sidebar: 
#     st.image('https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png')
#     st.title('LipBuddy')
#     st.info('This application is originally developed from the LipNet deep learning model.')

st.title('Lip Reading and Sentiment Analysis') 
# Generating a list of options or videos 
options = os.listdir(os.path.join('..', 'data', 's1'))
selected_video = st.selectbox('Choose video', options)

# Generate two columns 
col1, col2 = st.columns(2)

if options: 

    # Rendering the video 
    with col1: 
        st.info('The video below displays the converted video in mp4 format')
        file_path = os.path.join('..','data','s1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')

        # Rendering inside of the app
        video = open('test_video.mp4', 'rb') 
        video_bytes = video.read() 
        st.video(video_bytes)


    with col2: 
        st.info('System Health')
        video, annotations = load_data(tf.convert_to_tensor(file_path))
        
        # Convert float32 to uint8 and squeeze the grayscale channel dimension
        converted_video = np.clip(video * 255, 0, 255).astype(np.uint8)
        converted_video = np.squeeze(converted_video, axis=-1)  # Remove last dimension
        
        imageio.mimsave('animation.gif', converted_video, fps=10)
        st.image('animation.gif', width=400)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        
        
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)

        # ---- Sentiment Analysis (Lip Reading Output) ----
        st.info("Sentiment Analysis of Lip-Read Transcript")
        sentiment_result = analyze_sentiment(converted_prediction)

        if sentiment_result:
            st.success(
                f"Sentiment: {sentiment_result['sentiment']}  \n"
                f"Confidence: {sentiment_result['confidence']}"
            )
        else:
            st.warning("No sentiment detected.")

        
