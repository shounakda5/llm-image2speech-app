from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
import streamlit as st
import requests
import os

# Load environment variables
load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# img2txt
def image2text(image_path):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", max_new_tokens=100)
    text = image_to_text(image_path)[0]["generated_text"]
    print(text)
    return text

# LLM
def generate_story(scenario):
    template = f"""
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 150 words;
    
    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    # Initialize ChatOpenAI
    story_llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=1)

    # Create a RunnableSequence
    sequence = prompt | story_llm

    # Use the sequence to generate a story
    story = sequence.invoke({"scenario": scenario})
    story_text = story.content

    print(story_text)
    return story_text

# text to speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads = {
         "inputs": message
    }

    response = requests.post(API_URL, headers=headers, json=payloads)

    if response.status_code == 200:
        audio_path = os.path.join('data', 'audio.flac')
        with open(audio_path, 'wb') as file:
            file.write(response.content)
        print("Audio file saved successfully.")
        return audio_path
    else:
        print(f"Error: Unable to generate audio. Status code: {response.status_code}")
        return None

def main():
    st.set_page_config(page_title="Image to Audio Story", page_icon="🤖")

    st.header("Convert an image into an audio story")
    uploaded_file = st.file_uploader("Choose an image for your story", type="jpg")

    if uploaded_file is not None:
        print(uploaded_file)
        bytes_data = uploaded_file.getvalue()

        image_path = os.path.join('data', uploaded_file.name)
        with open(image_path, "wb") as file:
            file.write(bytes_data)

        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        scenario = image2text(image_path)
        story = generate_story(scenario)
        audio_path = text2speech(story)

        with st.expander("Scenario"):
            st.write(scenario)
        with st.expander("Story"):
            st.write(story)

        if audio_path:
            st.audio(audio_path)

if __name__ == '__main__':
    main()
