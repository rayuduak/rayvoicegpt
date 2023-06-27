import openai
import pyaudio
import pyttsx3
import constants
import speech_recognition as s
from rake_nltk import Rake

openai.api_type = constants.openai_api_type
openai.api_version = constants.openai_api_version
openai.api_key = constants.openai_api_key
openai.api_base = constants.openai_api_base

deployment_name = "gpt-35-turbo"
#initiate and set female voice
engine = pyttsx3.init()

try:
    engine.setProperty('voice',  engine.getProperty('voices')[1].id)
except:
    print("Setting engine voice failed")

# Please only use this one if you absolutely need it. It's slower and more expensive.
# deployment_name = "gpt-4"
# deployment_name = "gpt-4-32k"

# For embeddings only, but small private models may perform better and cheaper
# https://huggingface.co/spaces/mteb/leaderboard
# deployment_name = "text-embedding-ada-002"

#message = "Capital of Canada? Do not Elaborate"

# Read FAQ
# open text file in read mode
faq_file = open("QnA.txt", "r", encoding="utf8")
# read whole file to a string
context_data = faq_file.read()
# close file
faq_file.close()


# audio speech
sr = s.Recognizer()
with s.Microphone() as m:
    print("Welcome to ray gpt, your AI assistant, How can I help you?")
    engine.say("Welcome to ray gpt, your AI assistant, How can I help you?")
    engine.runAndWait()
    #exit(100)
    audio = sr.listen(m)
    audio_message = sr.recognize_google(audio)

if len(audio_message) < 5:
    exit(1)
print(audio_message)

# Extraction given the text using Rake
r = Rake()
r.extract_keywords_from_text(audio_message)
# To get keyword phrases ranked highest to lowest.
print(r.get_ranked_phrases())

# To get keyword phrases ranked highest to lowest with scores.
#print(r.get_ranked_phrases_with_scores())

# split and search
keywords = r.get_ranked_phrases()[0].split()
lines = context_data.split('\n')

# Iterate through each line and check for keywords
matching_lines = []
for line in lines:
    if any(keyword in line.lower() for keyword in keywords):
        matching_lines.append(line)

context_data = ' '.join(matching_lines)
print("---------------- context_data START-----------------")
print(context_data)
print("---------------- context_data END -----------------")

#exit(101)

prompt_msg = [
    {"role": "system", "content": "You are an AI Voice assistant. Act as Digital Assistant. " + context_data },
    {"role": "user", "content": audio_message + str("Instructions: 1)Please be concise and straight to the point unless asked to elaborate.2)If more answers appear Ask to be precise. 3)Do not tell Cell # .4) Tell only firstnames if any names found.5) if any website url, just convey visit website. DO NOT speak the url")}
    ]

# call openai to get response
response = openai.ChatCompletion.create(
    engine=deployment_name,
    temperature=0,
    messages=prompt_msg
)

print("--- token count---")
print(response['usage'].prompt_tokens)
print(response['usage'].completion_tokens)
print(response['usage'].total_tokens)


answer = response['choices'][0]['message']['content']
print("---genai response---")
print(answer)
engine.say(answer)
engine.runAndWait()
