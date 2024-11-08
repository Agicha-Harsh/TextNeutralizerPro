from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai
from fastapi import Request
import os
from dotenv import load_dotenv

app = FastAPI()

# Load the tokenizer and model for toxicity classification
tokenizer_toxicity = AutoTokenizer.from_pretrained("unitary/toxic-bert")
model_toxicity = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

# Load the tokenizer and model for sentiment analysis
tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Categories for toxicity classification
toxicity_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

load_dotenv()  # Load variables from .env
api_key = os.getenv("OPENAIPROJECT_API_KEY")

print("what is the API key we are getting over here",api_key)
# Initialize OpenAI client
client = openai.OpenAI(
    api_key=api_key,
    base_url="https://api.sambanova.ai/v1",
)

# Set up templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Function to classify the toxicity of a text
def classify_toxicity(text):
    inputs = tokenizer_toxicity(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model_toxicity(**inputs)
    logits = outputs.logits
    probs = torch.sigmoid(logits)  # Sigmoid because it's a multi-label classification
    predictions = (probs > 0.5).int()  # Threshold of 0.5 for classification
    return {label: predictions[0][i].item() for i, label in enumerate(toxicity_labels) if predictions[0][i].item() == 1}

# Function to determine the sentiment of a text
def get_sentiment(text):
    inputs = tokenizer_sentiment(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    outputs = model_sentiment(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)
    sentiment_labels = ["negative", "neutral", "positive"]
    predicted_class = torch.argmax(probs).item()
    return sentiment_labels[predicted_class]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/analyze/")
async def analyze_text(user_input: str = Form(...)):
    # Determine the sentiment of the sentence
    sentence_sentiment = get_sentiment(user_input)

    # Initialize the categorized words dictionary
    categorized_words = {label: [] for label in toxicity_labels}
    general_labels = []

    # Perform categorization only if the sentiment is negative
    if sentence_sentiment == "negative":
        # Tokenize the input sentence into words
        words = user_input.split()

        # Analyze toxicity of each word
        for word in words:
            toxicity = classify_toxicity(word)
            for label, is_present in toxicity.items():
                if is_present:
                    categorized_words[label].append(word)
                    general_labels.append(word)

        unique_general_labels = list(set(general_labels))
        sanitized_sentence = user_input
        for phrase in unique_general_labels:
            sanitized_sentence = sanitized_sentence.replace(phrase, "[profanity]")

        # Get rephrased sentences from OpenAI
        response = client.chat.completions.create(
            model='Meta-Llama-3.1-8B-Instruct',
            messages=[{"role": "user", "content": f"Rephrase the sentence to be more neutral, without losing the meaning behind it and give me atleast 3 proper response in plain text form, also don't give the numbering list where each reponse should cover all the input content: {sanitized_sentence}"}],
            temperature=0.5,
            top_p=0.5
        )

        response_content = response.choices[0].message.content

        # Extract relevant lines (without numbering)
        # Extract lines that start with digits (the numbered list)
        extracted_list = [line.strip() for line in response_content.split("\n") if line.strip() and not line.startswith("Here are")]

        # If no numbered list is found, use a fallback to extract important sentences
        # if not extracted_list:
        #     print("\nNo numbered list found. Extracting all relevant sentences instead:")
        #     extracted_list = [line.strip() for line in response_content.split('\n') if line.strip()]
        print("Output of the API",extracted_list)
        return {"extracted_list": extracted_list, "original_sentence": user_input, "sanitized_sentence":sanitized_sentence}

    else:
        # Sweet alert for non-negative sentences
        return JSONResponse(content={"message": "The sentiment is positive. Text posted successfully."})

@app.post("/submit/")
async def submit_text(selected_sentence: str = Form(...)):
    # Determine the sentiment of the selected sentence
    sentence_sentiment = get_sentiment(selected_sentence)

    # Check if the sentiment is still negative
    # if sentence_sentiment == "negative":
    #     return {"message": "The selected sentence still has a negative sentiment. Please try again."}

    # If the sentiment is not negative, submit the text
    return {"message": "Text selected successfully."}
