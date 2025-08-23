import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Load model once
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

device = 0 if torch.cuda.is_available() else -1

sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model=model,
    tokenizer=tokenizer,
    device=device
)

# Function to add sentiment to a DataFrame
# Function to add sentiment to a DataFrame (batch processing)
# Function to add sentiment to a DataFrame
def add_sentiment(df, batch_size=32):
    """
    Add sentiment and confidence columns to a DataFrame with a 'message' column.
    Uses batch processing for efficiency.
    """

    # Make a safe copy of the DataFrame (to avoid modifying the original one directly)
    df_copy = df.copy()

    # Convert all messages to strings and cut them to max 512 tokens (model limit)
    texts = [str(x)[:512] for x in df_copy["message"]]

    results = []  # this will hold (label, score) for each message

    # Process messages in batches instead of one-by-one → much faster
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]        # take one batch of messages
        outputs = sentiment_pipeline(batch)      # run model on the batch
        results.extend([(o["label"], o["score"]) for o in outputs])  # collect results

    # Convert results list → two new DataFrame columns: sentiment + confidence
    df_copy[["sentiment", "confidence"]] = pd.DataFrame(results, index=df_copy.index)

    return df_copy


# Function to compute user similarity based on sentiment
def user_sentiment_similarity(df):
    """Compute cosine similarity matrix between users based on sentiment counts."""
    user_sentiment = pd.crosstab(df["user"], df["sentiment"])
    user_sentiment_norm = user_sentiment.div(user_sentiment.sum(axis=1), axis=0).fillna(0)
    similarity_matrix = cosine_similarity(user_sentiment_norm)
    sim_df = pd.DataFrame(similarity_matrix, index=user_sentiment_norm.index, columns=user_sentiment_norm.index)
    return sim_df, user_sentiment
