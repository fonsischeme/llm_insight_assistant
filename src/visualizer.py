import plotly.express as px
import pandas as pd
from typing import List, Dict

def bar_top_themes(theme_counts: Dict[str, int]):
    df = pd.DataFrame({"theme": list(theme_counts.keys()), "count": list(theme_counts.values())})
    fig = px.bar(df, x="theme", y="count", title="Top Themes")
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    return fig

def pie_sentiment(sentiment_counts: Dict[str, int]):
    df = pd.DataFrame({"sentiment": list(sentiment_counts.keys()), "count": list(sentiment_counts.values())})
    fig = px.pie(df, names="sentiment", values="count", title="Sentiment Distribution")
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(margin=dict(l=10,r=10,t=40,b=10))
    return fig
