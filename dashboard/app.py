"""
dashboard/app.py
─────────────────
Phase 9: Streamlit Visual Dashboard

HOW TO RUN:
    streamlit run dashboard/app.py

FEATURES:
  1. Upload articles manually OR use mock data
  2. View framing comparison across sources
  3. Sentiment bar chart
  4. Bias score comparison
  5. Entity prominence heatmap
  6. Per-article breakdown table
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFIE — Narrative Framing Intelligence Engine",
    page_icon="N",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #e63946, #457b9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: #1e1e2e;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #313244;
    }
    .divergence-extreme { color: #e63946; font-weight: bold; }
    .divergence-high    { color: #f4a261; font-weight: bold; }
    .divergence-moderate{ color: #e9c46a; }
    .divergence-low     { color: #a8dadc; }
    .divergence-minimal { color: #57cc99; }
    .source-tag {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.8rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────

def run_pipeline(articles: list[dict]) -> dict:
    """Run the full NFIE analysis pipeline."""
    from src.comparison.engine import run_full_pipeline
    return run_full_pipeline(articles)


def divergence_color(level: str) -> str:
    colors = {
        "Extreme": "#e63946",
        "High": "#f4a261",
        "Moderate": "#e9c46a",
        "Low": "#a8dadc",
        "Minimal": "#57cc99",
    }
    return colors.get(level, "#888")


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## NFIE")
    st.markdown("**Narrative Framing Intelligence Engine**")
    st.divider()

    input_mode = st.radio(
        "Data Source",
        ["Use Mock Dataset", "Paste Articles", "NewsAPI Query"],
        index=0,
    )

    st.divider()
    st.markdown("### Analysis Options")
    run_ner = st.toggle("Named Entity Recognition", value=True)
    run_frames = st.toggle("Frame Classification", value=True)
    run_sentiment = st.toggle("Sentiment Analysis", value=True)
    run_bias = st.toggle("Bias Detection", value=True)

    st.divider()
    st.caption("Built with Transformers, spaCy, FAISS, and FastAPI")


# ── Main Title ─────────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">Narrative Framing Intelligence Engine</div>', unsafe_allow_html=True)
st.markdown("*Revealing how different news outlets frame the same event*")
st.divider()


# ── Input Section ──────────────────────────────────────────────────────────────

articles = []

if input_mode == "Use Mock Dataset":
    st.info("Using built-in mock dataset: **US-China Trade War (Tariffs)**  \n"
            "8 articles from: Reuters, Fox News, The Guardian, Breitbart, Al Jazeera, NPR, Wired, Foreign Affairs")

    if st.button("Run Analysis", type="primary", use_container_width=True):
        from src.ingestion.fetcher import get_mock_articles
        articles = get_mock_articles("US China tariff")
        st.session_state["articles"] = articles
        st.session_state["run_analysis"] = True

elif input_mode == "Paste Articles":
    st.markdown("### Add Articles")
    num_articles = st.slider("Number of articles to compare", 2, 8, 3)

    form_articles = []
    for i in range(num_articles):
        with st.expander(f"Article {i+1}", expanded=(i == 0)):
            col1, col2 = st.columns(2)
            title = col1.text_input(f"Title", key=f"title_{i}", placeholder="Article headline")
            source = col2.text_input(f"Source", key=f"source_{i}", placeholder="e.g., Reuters")
            content = st.text_area(f"Content", key=f"content_{i}", height=120,
                                   placeholder="Paste article body here...")
            if title and source and content:
                from src.ingestion.fetcher import normalize_article
                form_articles.append(normalize_article(
                    title=title, content=content, source=source,
                    url="", date="", query="manual",
                ))

    if len(form_articles) >= 2:
        if st.button("Run Analysis", type="primary", use_container_width=True):
            articles = form_articles
            st.session_state["articles"] = articles
            st.session_state["run_analysis"] = True
    else:
        st.warning("Fill in at least 2 articles to enable comparison")

elif input_mode == "NewsAPI Query":
    query = st.text_input("Search query", placeholder="e.g., US China trade war")
    use_mock = st.checkbox("Fall back to mock if no API key", value=True)

    if query and st.button("Fetch and Analyze", type="primary", use_container_width=True):
        from src.ingestion.fetcher import NewsAPIFetcher, get_mock_articles
        with st.spinner("Fetching articles..."):
            if use_mock:
                articles = get_mock_articles(query)
            else:
                fetcher = NewsAPIFetcher()
                articles = fetcher.fetch(query, page_size=10)
        if articles:
            st.session_state["articles"] = articles
            st.session_state["run_analysis"] = True
        else:
            st.error("No articles found. Try a different query.")


# ── Run Pipeline ───────────────────────────────────────────────────────────────

if st.session_state.get("run_analysis") and st.session_state.get("articles"):
    articles = st.session_state["articles"]
    st.session_state["run_analysis"] = False  # Reset flag

    with st.spinner("Running analysis pipeline... (this may take 30-60 seconds on first run)"):
        try:
            report = run_pipeline(articles)
            st.session_state["report"] = report
            st.success(f"Analysis complete for {len(articles)} articles")
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.exception(e)


# ── Results ────────────────────────────────────────────────────────────────────

if "report" in st.session_state:
    report = st.session_state["report"]

    # ── Top Metrics Row ──────────────────────────────────────────────────────
    st.markdown("## Analysis Results")
    col1, col2, col3, col4 = st.columns(4)

    divergence = report.get("overall_divergence_score", 0)
    div_level = report.get("divergence_level", "Unknown")
    num_articles = report.get("articles_analyzed", 0)
    sources = report.get("sources", [])

    col1.metric("Articles Analyzed", num_articles)
    col2.metric("Sources", len(set(sources)))
    col3.metric("Overall Divergence", f"{divergence:.2f}")

    div_color = divergence_color(div_level)
    col4.markdown(
        f"**Divergence Level**  \n"
        f'<span style="color:{div_color}; font-size:1.3rem; font-weight:bold">{div_level}</span>',
        unsafe_allow_html=True,
    )

    st.divider()

    # ── Key Differences ──────────────────────────────────────────────────────
    key_diffs = report.get("key_differences", [])
    if key_diffs:
        st.markdown("### Key Framing Differences")
        for diff in key_diffs:
            label, _, rest = diff.partition(": ")
            st.markdown(f"**{label}:** {rest}")
        st.divider()

    # ── Two Column Layout ────────────────────────────────────────────────────
    left_col, right_col = st.columns(2)

    # ── Sentiment Chart ──────────────────────────────────────────────────────
    with left_col:
        st.markdown("### Sentiment by Source")
        sentiment_data = report.get("sentiment_comparison", {}).get("sentiment_by_source", {})
        if sentiment_data:
            sent_df = pd.DataFrame([
                {
                    "Source": source,
                    "Compound Score": data.get("compound", 0),
                    "Label": data.get("label", "neutral").title(),
                    "Tone": data.get("tone", "").title(),
                }
                for source, data in sentiment_data.items()
            ])

            color_map = {
                "Positive": "#57cc99",
                "Neutral": "#a8dadc",
                "Negative": "#e63946",
            }

            fig = px.bar(
                sent_df,
                x="Source",
                y="Compound Score",
                color="Label",
                color_discrete_map=color_map,
                title="Sentiment Compound Score (-1 = most negative, +1 = most positive)",
            )
            fig.update_layout(
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="white",
                showlegend=True,
                height=350,
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
            st.plotly_chart(fig, use_container_width=True)

    # ── Bias Score Chart ─────────────────────────────────────────────────────
    with right_col:
        st.markdown("### Bias Scores by Source")
        bias_data = report.get("bias_comparison", {}).get("bias_scores", {})
        if bias_data:
            bias_df = pd.DataFrame([
                {"Source": source, "Bias Score": score}
                for source, score in bias_data.items()
            ]).sort_values("Bias Score", ascending=True)

            fig = px.bar(
                bias_df,
                x="Bias Score",
                y="Source",
                orientation="h",
                color="Bias Score",
                color_continuous_scale="RdYlGn_r",
                title="Lexical Bias Score (0 = neutral, 1 = highly loaded)",
            )
            fig.update_layout(
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117",
                font_color="white",
                height=350,
                coloraxis_showscale=False,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Frame Comparison Radar ────────────────────────────────────────────────
    st.markdown("### Frame Distribution")
    frames_by_source = report.get("frame_comparison", {}).get("frames_by_source", {})
    if frames_by_source:
        frame_labels = [
            "political", "economic", "emotional",
            "security", "nationalist", "humanitarian", "legal", "scientific"
        ]

        fig = go.Figure()
        colors = ["#e63946", "#457b9d", "#2a9d8f", "#e9c46a", "#f4a261", "#a8dadc", "#e76f51", "#264653"]

        for i, (source, data) in enumerate(frames_by_source.items()):
            all_scores = data.get("all_scores", {})
            values = [all_scores.get(f, 0) for f in frame_labels]
            values_closed = values + [values[0]]  # Close the polygon

            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=frame_labels + [frame_labels[0]],
                fill="toself",
                name=source,
                opacity=0.6,
                line_color=colors[i % len(colors)],
            ))

        fig.update_layout(
            polar=dict(
                bgcolor="#1e1e2e",
                radialaxis=dict(visible=True, range=[0, 1], color="gray"),
                angularaxis=dict(color="white"),
            ),
            paper_bgcolor="#0e1117",
            font_color="white",
            height=450,
            showlegend=True,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Per-Article Summary Table ─────────────────────────────────────────────
    st.markdown("### Per-Article Summary")
    per_article = report.get("per_article_summary", [])
    if per_article:
        df = pd.DataFrame(per_article)

        # Style the dataframe
        display_cols = [
            "source", "title", "primary_frame",
            "sentiment_label", "dominant_tone", "bias_score"
        ]
        df_display = df[display_cols].rename(columns={
            "source": "Source",
            "title": "Headline",
            "primary_frame": "Primary Frame",
            "sentiment_label": "Sentiment",
            "dominant_tone": "Tone",
            "bias_score": "Bias",
        })

        st.dataframe(
            df_display,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Bias": st.column_config.ProgressColumn(
                    "Bias Score",
                    help="0 = neutral, 1 = highly loaded",
                    format="%.2f",
                    min_value=0,
                    max_value=1,
                ),
            }
        )

    # ── Entity Comparison ─────────────────────────────────────────────────────
    st.markdown("### Entity Prominence by Source")
    entity_comp = report.get("entity_comparison", {}).get("entity_comparison", [])
    if entity_comp:
        top_entities = entity_comp[:10]
        sources_list = list(set(sources))

        heatmap_data = []
        for ent_data in top_entities:
            row = {"Entity": ent_data["entity"]}
            for src in sources_list:
                row[src] = ent_data["mentions_by_source"].get(src, 0)
            heatmap_data.append(row)

        if heatmap_data:
            hm_df = pd.DataFrame(heatmap_data).set_index("Entity")
            fig = px.imshow(
                hm_df,
                color_continuous_scale="Blues",
                title="Entity Mention Count by Source (darker = more prominent)",
                aspect="auto",
            )
            fig.update_layout(
                paper_bgcolor="#0e1117",
                font_color="white",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── Raw JSON (for developers) ─────────────────────────────────────────────
    with st.expander("Raw Analysis JSON"):
        # Show report without enriched_articles (too verbose)
        display_report = {k: v for k, v in report.items() if k != "enriched_articles"}
        st.json(display_report)

else:
    # Welcome screen
    st.markdown("## Select a data source in the sidebar to begin")
    st.markdown("""
    ### What NFIE Does

    The Narrative Framing Intelligence Engine reveals **how different news outlets
    construct reality** when covering the same event.

    **It detects:**

    | Signal | What It Reveals |
    |--------|----------------|
    | Framing | Political, economic, nationalist, or emotional angles |
    | Sentiment | Emotional charge: negative, neutral, or positive |
    | Lexical Bias | Loaded words, power vocabulary, alarm language |
    | Entity Prominence | Who and what is emphasized per outlet |
    | Divergence Score | How differently sources framed the event |

    **Select "Use Mock Dataset" in the sidebar and click "Run Analysis" to get started.**
    """)
