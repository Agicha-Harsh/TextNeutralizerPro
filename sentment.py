from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import openai

# Load the tokenizer and model for toxicity classification
tokenizer_toxicity = AutoTokenizer.from_pretrained("unitary/toxic-bert")
model_toxicity = AutoModelForSequenceClassification.from_pretrained("unitary/toxic-bert")

# Load the tokenizer and model for sentiment analysis
tokenizer_sentiment = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
model_sentiment = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

# Categories for toxicity classification
toxicity_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

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

# Get user input for the sentence
user_input = input("Enter a sentence for analysis: ")

# Debugging print to check user input
#print(f"Debug: User input received: '{user_input}'")
print("User input: " + user_input)
# Determine the sentiment of the sentence
sentence_sentiment = get_sentiment(user_input)
print("Sentence sentiment detected as: "+ sentence_sentiment)

# Initialize the categorized words dictionary
categorized_words = {label: [] for label in toxicity_labels}
general_labels = []
# Perform categorization only if the sentiment is negative
if sentence_sentiment == "negative":
    # Tokenize the input sentence into words
    words = user_input.split()

    # Debugging print to check tokenized words
    print("Tokenized words: ", words)

    # Analyze toxicity of each word
    for word in words:
        toxicity = classify_toxicity(word)
        for label, is_present in toxicity.items():
            if is_present:
                categorized_words[label].append(word)
                general_labels.append(word)

    print("general _lable list",general_labels)
    # Display results
    for label in toxicity_labels:
        if categorized_words[label]:
            print("Words classified as " + label + ": " , categorized_words[label])
        else:
            print("No words classified as " + label + " found.")
    client = openai.OpenAI(
        api_key="a6f5f579-ffdd-41f7-bbe3-1676f33fcf20",
        base_url="https://api.sambanova.ai/v1",
    )
    # toxic_sentence = "FUCK YOUR FILTHY MOTHER IN THE ASS, DRY!"
    # user_input = "Fuck you, block me, you faggot pussy!"
    # general_labels = ['Fuck', 'Fuck', 'you,', 'faggot', 'faggot', 'faggot', 'faggot', 'pussy!', 'pussy!']
    unique_general_labels = list(set(general_labels))
    sanitized_sentence = user_input
    for phrase in unique_general_labels:
        sanitized_sentence = sanitized_sentence.replace(phrase, "[profanity]")
        # print("sanitized sentence",sanitized_sentence)
    # sanitized_sentence = user_input.replace("FUCK", "[profanity]").replace("FILTHY", "[profanity]").replace("ASS","[profanity]")

    print("what is the sanitized sentence",sanitized_sentence)
    response = client.chat.completions.create(
        model='Meta-Llama-3.1-8B-Instruct',
        messages=[{"role": "user", "content": f"Rephrase the sentence to be more neutral, without losing the meaning behind it and give me atleast 3 proper response in plain text form, also don't give the numbering list where each reponse should cover all the input content: {sanitized_sentence}"}],
        temperature=0.5,
        top_p=0.5
    )

    response_content = response.choices[0].message.content

    # Print the raw response content for debugging purposes
    print("\nResponse content: ", response_content)

    # Extract lines that start with digits (the numbered list)
    extracted_list = [line.strip() for line in response_content.split("\n") if line.strip() and not line.startswith("Here are")]

    # If no numbered list is found, use a fallback to extract important sentences
    if not extracted_list:
        print("\nNo numbered list found. Extracting all relevant sentences instead:")
        extracted_list = [line.strip() for line in response_content.split('\n') if line.strip()]

    # Print the extracted list of polite alternatives
    print("\nExtracted list of polite alternatives:")
    for phrase in extracted_list:
        print(phrase)

else:
    print("The sentiment of the sentence is not negative. No categorization performed.")
