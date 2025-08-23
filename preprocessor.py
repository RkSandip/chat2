import re
import pandas as pd

def preprocess(data: str) -> pd.DataFrame:
    """
    Preprocess WhatsApp chat text into a structured DataFrame.
    Handles 2-digit/4-digit years and 12/24-hour time.
    """

    # Regex: match date, time, and the rest of message
    # Example matches:
    #   21/08/25, 22:16 - User: msg
    #   21/08/2025, 10:45 AM - User: msg
    pattern = r'(\d{1,2}/\d{1,2}/\d{2,4}),\s(\d{1,2}:\d{2}(?:\s?[APMapm]{2})?)\s-\s'

    # Split into messages
    messages = re.split(pattern, data)
    if not messages[0].strip():  # drop empty first split
        messages = messages[1:]

    dates, texts = [], []

    for i in range(0, len(messages), 3):
        try:
            date_str = f"{messages[i]},{messages[i+1]}"
            msg = messages[i+2]
            dates.append(date_str.strip())
            texts.append(msg.strip())
        except IndexError:
            continue

    df = pd.DataFrame({'user_message': texts, 'message_date': dates})

    # Flexible datetime parser
    def parse_date(x):
        x = x.strip()
        date_formats = [
            "%d/%m/%Y,%H:%M",   # 21/08/2025,22:16
            "%d/%m/%y,%H:%M",   # 21/08/25,22:16
            "%d/%m/%Y,%I:%M %p",# 21/08/2025,10:45 PM
            "%d/%m/%y,%I:%M %p" # 21/08/25,10:45 PM
        ]
        for fmt in date_formats:
            try:
                dt = pd.to_datetime(x, format=fmt)
                if dt.year < 100:  # adjust weird 2-digit years
                    dt = dt.replace(year=dt.year + 2000)
                return dt
            except:
                continue
        return pd.NaT

    df['date'] = df['message_date'].apply(parse_date)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')  # final safeguard
    df.drop(columns=['message_date'], inplace=True)

    # Split user and message
    users, messages_clean = [], []
    for msg in df['user_message']:
        entry = re.split(r'([\w\W]+?):\s', msg, maxsplit=1)
        if len(entry) >= 3:
            users.append(entry[1])
            messages_clean.append(entry[2])
        else:
            users.append('group_notification')
            messages_clean.append(entry[0])

    df['user'] = users
    df['message'] = messages_clean
    df.drop(columns=['user_message'], inplace=True)

    # Remove unwanted messages
    df['message'] = df['message'].astype(str).str.strip()
    remove_list = [
        "<Media omitted>",
        "This message was deleted",
        "You deleted this message"
    ]
    df = df[~df["message"].isin(remove_list)]
    df = df[df["message"] != ""].reset_index(drop=True)

    # Extra datetime columns
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    # Period column for heatmaps
    period = []
    for hour in df['hour']:
        if pd.isna(hour):
            period.append(None)
        elif hour == 23:
            period.append(f"{hour}-00")
        elif hour == 0:
            period.append(f"00-{hour+1}")
        else:
            period.append(f"{hour}-{hour+1}")
    df['period'] = period

    return df
