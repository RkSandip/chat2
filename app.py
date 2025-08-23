import streamlit as st

# ✅ Set page config 
st.set_page_config(
    # layout="wide"
    page_title= "ChatSpectrum",
    page_icon= "icon.png"
    )


import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import sentiment  # your sentiment.py
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import hashlib
import os

plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Color Emoji", "Nirmala UI"]

# ===================== Custom Background Color ===================== #bffcc6
st.markdown(
    """
    <style>
    /* Main app background */
    .stApp {
        background-color: #159688;
    }

    /* Main content text only */
    .block-container {
        color: black !important;
    }

    /* Sidebar background and text remain default */
    .css-1d391kg {  /* sidebar container class in Streamlit 1.x/2.x */
        color: initial !important;
    }

    /* Keep top menu buttons visible */
    button[title="Settings"], button[title="Send feedback"], .css-1v3fvcr { 
        color: initial !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ===================== ✅ Caching =====================
if "file_hash" not in st.session_state:
    st.session_state.file_hash = None
if "user_analysis_cache" not in st.session_state:
    st.session_state.user_analysis_cache = {}
if "sentiment_done" not in st.session_state:
    st.session_state.sentiment_done = False
    st.session_state.sentiment_result = None
# =======================================================


st.title("Whatsapp Chat Analyzer")

# Path to default file
default_file_path = "demo.txt"



# ===================== File Upload / Default =====================
uploaded_file = st.sidebar.file_uploader("Choose a text file", type="txt")

if uploaded_file is not None:
    # Clean the filename on the fly
    filename = uploaded_file.name.replace(" ", "_").replace(",", "_")
    st.markdown(f"**File Name:** - &nbsp; {filename}", unsafe_allow_html=True)

    # Read file content
    data_bytes = uploaded_file.getvalue()
    data = data_bytes.decode("utf-8")

elif os.path.exists(default_file_path):
    # Use default file
    st.markdown(f"**File Name:** - &nbsp; {default_file_path} (default)", unsafe_allow_html=True)
    with open(default_file_path, "r", encoding="utf-8") as f:
        data = f.read()

else:
    st.error(f"Default file '{default_file_path}' not found.")
    st.stop()


# ===================== Compute hash for caching =====================
file_hash = hashlib.md5(data.encode("utf-8")).hexdigest()
if st.session_state.file_hash != file_hash:
    st.session_state.file_hash = file_hash
    st.session_state.user_analysis_cache = {}
    st.session_state.sentiment_done = False
    st.session_state.sentiment_result = None

# ===================== Preprocess data =====================
df = preprocessor.preprocess(data)

# ===================== Fetch users =====================
user_list = df['user'].unique().tolist()
if "group_notification" in user_list:
    user_list.remove('group_notification')
user_list.sort()
user_list.insert(0, "Overall")

# ===================== Mode selection =====================
mode = st.sidebar.radio("Choose Mode", ["Show Analysis", "Sentiment Analysis"])

# ===================== SHOW ANALYSIS =====================
if mode == "Show Analysis":
    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)

    # Check if analysis for this user is already cached
    if selected_user not in st.session_state.user_analysis_cache:
        # Start computation for this user
        st.session_state.user_analysis_cache[selected_user] = {}  # store results in a dict
        cache = st.session_state.user_analysis_cache[selected_user]

        # Stats
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        cache['stats'] = (num_messages, words, num_media_messages, num_links)

        # Timelines
        cache['monthly'] = helper.monthly_timeline(selected_user, df)
        cache['daily'] = helper.daily_timeline(selected_user, df)

        # Activity maps
        cache['busy_day'] = helper.week_activity_map(selected_user, df)
        cache['busy_month'] = helper.month_activity_map(selected_user, df)
        cache['heatmap'] = helper.activity_heatmap(selected_user, df)

        # Most busy users (only Overall)
        if selected_user == "Overall":
            df_users = df[df["user"] != "group_notification"]
            x, new_df = helper.most_busy_users(df_users)
            cache['most_busy_users'] = (x, new_df)

        # WordCloud
        cache['wordcloud'] = helper.create_wordcloud(selected_user, df)

        # Most common words
        cache['common_words'] = helper.most_common_words(selected_user, df)

        # Emoji analysis
        cache['emoji'] = helper.emoji_helper(selected_user, df)

    # Use cached data
    cache = st.session_state.user_analysis_cache[selected_user]

    # Display Stats
    num_messages, words, num_media_messages, num_links = cache['stats']
    st.title("Top Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.header("Total Messages"); st.title(num_messages)
    with col2: st.header("Total Words"); st.title(words)
    with col3: st.header("Media Shared"); st.title(num_media_messages)
    with col4: st.header("Links Shared"); st.title(num_links)

    # Monthly Timeline
    st.title("Monthly Timeline")
    timeline = cache['monthly']
    fig, ax = plt.subplots()
    ax.plot(timeline['time'], timeline['message'], color='green')
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

    # Daily Timeline
    st.title("Daily Timeline")
    daily_timeline = cache['daily']
    fig, ax = plt.subplots()
    ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
    plt.xticks(rotation='vertical')
    st.pyplot(fig)

    # Activity Map
    st.title("Activity Map")
    col1, col2 = st.columns(2)
    with col1:
        st.header("Most busy day")
        busy_day = cache['busy_day']
        fig, ax = plt.subplots()
        ax.bar(busy_day.index, busy_day.values, color='purple')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)
    with col2:
        st.header("Most busy month")
        busy_month = cache['busy_month']
        fig, ax = plt.subplots()
        ax.bar(busy_month.index, busy_month.values, color='orange')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

    st.title("Weekly Activity Map")
    user_heatmap = cache['heatmap']
    fig, ax = plt.subplots()
    ax = sns.heatmap(user_heatmap)
    st.pyplot(fig)

    # Most Busy Users
    if selected_user == "Overall":
        st.title("Most Busy Users")
        x, new_df = cache['most_busy_users']
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.bar(x.index, x.values, color='red')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        with col2:
            st.dataframe(new_df)

    # WordCloud
    st.title("Wordcloud")
    df_wc = cache['wordcloud']
    fig, ax = plt.subplots()
    ax.imshow(df_wc)
    st.pyplot(fig)

    # Most common words
    most_common_df = cache['common_words']
    fig, ax = plt.subplots()
    ax.barh(most_common_df[0], most_common_df[1])
    plt.xticks(rotation='vertical')
    st.title("Most common words")
    st.pyplot(fig)

    # Emoji analysis
    emoji_df = cache['emoji']
    st.title("Emoji Analysis")
    col1, col2 = st.columns(2)
    with col1:
        if emoji_df.empty:
            st.write("No emojis found for the selected user.")
        else:
            st.dataframe(emoji_df)
    with col2:
        # plt.rcParams['font.family'] = 'Segoe UI Emoji'
        if emoji_df.empty:
            st.write("No emojis to display.")
        else:
            fig, ax = plt.subplots()
            ax.pie(emoji_df['count'].head(), labels=emoji_df['emoji'].head(), autopct="%0.2f")
            st.pyplot(fig)

# ===================== SENTIMENT ANALYSIS =====================
if mode == "Sentiment Analysis":
    if not st.session_state.sentiment_done:
        df_sent = sentiment.add_sentiment(df)
        st.session_state.sentiment_result = df_sent
        st.session_state.sentiment_done = True

    st.subheader("Sentiment Analysis")
    df_sent = st.session_state.sentiment_result
    df_sent = df_sent[df_sent["user"] != "group_notification"]

    # Overall Sentiment
    sentiment_counts = df_sent["sentiment"].value_counts()
    colors = {"positive": "green", "negative": "red", "neutral": "grey"}
    st.write("### Overall Sentiment Distribution")
    fig, ax = plt.subplots(figsize=(6,4))
    sentiment_counts.plot(kind="bar", color=[colors.get(x,"blue") for x in sentiment_counts.index], ax=ax)
    ax.set_title("Overall Sentiment Distribution")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Message Count")
    ax.set_xticks(range(len(sentiment_counts.index)))
    ax.set_xticklabels(sentiment_counts.index, rotation=0)
    st.pyplot(fig)

    # User similarity heatmap (Top 20)
    user_sentiment = pd.crosstab(df_sent["user"], df_sent["sentiment"])
    N = 20
    top_users = df_sent["user"].value_counts().head(N).index
    user_sentiment_norm = user_sentiment.div(user_sentiment.sum(axis=1), axis=0).fillna(0)
    similarity_matrix = cosine_similarity(user_sentiment_norm)
    sim_df = pd.DataFrame(similarity_matrix, index=user_sentiment_norm.index, columns=user_sentiment_norm.index)
    sim_df_top = sim_df.loc[top_users, top_users]
    sim_df_top.index.name = None
    sim_df_top.columns.name = None
    mask = np.triu(np.ones_like(sim_df_top, dtype=bool), k=1)
    st.write("### User Sentiment Similarity Heatmap (Top 20 Users)")
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(sim_df_top, mask=mask, cmap="GnBu", annot=True, fmt=".2f", annot_kws={"size":8}, ax=ax)
    st.pyplot(fig)

    # ========= STEP 6: Per-User Sentiment Breakdown (Top 50 Users) ==========
    user_sentiment = pd.crosstab(df_sent["user"], df_sent["sentiment"])
    top_users = df_sent["user"].value_counts().head(50).index
    user_sentiment_top = user_sentiment.loc[top_users]

    st.write("### Per-User Sentiment Distribution (Top 50 Users)")
    fig, ax = plt.subplots(figsize=(14,6))
    user_sentiment_top.plot(kind="bar", stacked=True,
                            color=[colors.get(x,"blue") for x in user_sentiment_top.columns], ax=ax)
    ax.set_xlabel("User")
    ax.set_ylabel("Message Count")
    ax.legend(title="Sentiment", bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout()
    st.pyplot(fig)

    # Top negative messages
    negative_df = df_sent[df_sent["sentiment"] == "negative"].sort_values(by="confidence", ascending=False)
    st.write("### Top Negative Messages (Top 100)")
    st.dataframe(negative_df[["user", "message", "confidence"]].head(100))
