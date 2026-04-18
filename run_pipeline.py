"""
scripts/run_pipeline.py
────────────────────────
CLI runner to test the entire NFIE pipeline end-to-end.

RUN:
    cd /home/claude/nfie
    python scripts/run_pipeline.py

This script exercises every phase in sequence and prints
a rich formatted report to the terminal.
"""

import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from rich.rule import Rule

console = Console()


def print_phase(num: int, name: str):
    console.print(Rule(f"[bold cyan]PHASE {num}: {name}[/bold cyan]"))


def run():
    console.print(Panel.fit(
        "[bold white]Narrative Framing Intelligence Engine[/bold white]\n"
        "[dim]Full Pipeline Test Run[/dim]",
        border_style="cyan",
    ))

    # ─────────────────────────────────────────────────────────────────
    # PHASE 1: DATA INGESTION
    # ─────────────────────────────────────────────────────────────────
    print_phase(1, "DATA INGESTION")

    from src.ingestion.fetcher import get_mock_articles, save_articles
    from src.utils.cache import BatchProcessor

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Loading mock dataset...", total=None)
        articles = get_mock_articles("US China tariff trade war")
        articles = BatchProcessor.deduplicate_articles(articles)
        articles = BatchProcessor.filter_short_articles(articles, min_words=30)
        progress.update(task, completed=True)

    save_articles(articles)

    table = Table(title=f"Loaded {len(articles)} Articles", show_lines=True)
    table.add_column("Source", style="cyan", width=20)
    table.add_column("Title", style="white", width=60)
    table.add_column("Words", style="green", justify="right")

    for a in articles:
        table.add_row(a["source"], a["title"][:58], str(a["word_count"]))

    console.print(table)

    # ─────────────────────────────────────────────────────────────────
    # PHASE 2: EMBEDDINGS & CLUSTERING
    # ─────────────────────────────────────────────────────────────────
    print_phase(2, "EMBEDDINGS & CLUSTERING (FAISS)")

    from src.clustering.embedder import build_index_and_clusters, save_clusters

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Building embeddings + FAISS index...", total=None)
        faiss_store, embeddings, clusters = build_index_and_clusters(articles, save_index=True)
        save_clusters(clusters)
        progress.update(task, completed=True)

    console.print(f"[green]✓[/green] Embedding shape: {embeddings.shape}")
    console.print(f"[green]✓[/green] FAISS index: {faiss_store.index.ntotal} vectors")
    console.print(f"[green]✓[/green] Clusters found: {len(clusters)}")

    for c in clusters[:3]:
        sources = ", ".join(c["sources"])
        console.print(f"  [dim]Cluster {c['cluster_id']}:[/dim] {c['size']} articles → {sources}")

    # Use largest cluster for downstream analysis
    target_cluster = clusters[0]
    cluster_articles = target_cluster["articles"]
    console.print(f"\n[bold]Using largest cluster:[/bold] {len(cluster_articles)} articles")

    # ─────────────────────────────────────────────────────────────────
    # PHASE 3: NAMED ENTITY RECOGNITION
    # ─────────────────────────────────────────────────────────────────
    print_phase(3, "NAMED ENTITY RECOGNITION")

    from src.ner.extractor import NERExtractor, validate_cluster_with_ner

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Running spaCy NER...", total=None)
        ner = NERExtractor()
        cluster_articles = ner.extract_batch(cluster_articles)
        target_cluster = validate_cluster_with_ner({**target_cluster, "articles": cluster_articles})
        progress.update(task, completed=True)

    coherence = target_cluster.get("entity_coherence_score", 0)
    shared = target_cluster.get("shared_entities", [])

    console.print(f"[green]✓[/green] Entity coherence score: [bold]{coherence:.3f}[/bold] "
                  f"({'Good cluster' if coherence > 0.3 else 'Weak cluster'})")
    console.print(f"[green]✓[/green] Shared entities: {', '.join(shared[:10])}")

    # Show entity table for first article
    if cluster_articles:
        a = cluster_articles[0]
        ec = a.get("entity_counts", {})
        ent_table = Table(title=f"NER: {a['source']}", show_lines=False)
        ent_table.add_column("Type", style="yellow")
        ent_table.add_column("Entities", style="white")
        for label, entities in list(ec.items())[:5]:
            top = ", ".join(f"{k}({v})" for k, v in list(entities.items())[:4])
            ent_table.add_row(label, top)
        console.print(ent_table)

    # ─────────────────────────────────────────────────────────────────
    # PHASE 4: FRAME CLASSIFICATION
    # ─────────────────────────────────────────────────────────────────
    print_phase(4, "FRAME CLASSIFICATION")

    from src.classification.framer import FrameClassifier, compare_frames

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Classifying frames (zero-shot)...", total=None)
        framer = FrameClassifier()
        cluster_articles = framer.classify_batch(cluster_articles)
        frame_comparison = compare_frames(cluster_articles)
        progress.update(task, completed=True)

    frame_table = Table(title="Frame Analysis by Source", show_lines=True)
    frame_table.add_column("Source", style="cyan", width=18)
    frame_table.add_column("Primary Frame", style="bold yellow", width=16)
    frame_table.add_column("Top 3 Frames", style="white", width=45)

    for source, data in frame_comparison.get("frames_by_source", {}).items():
        top_frames = " | ".join(
            f"{f['frame']}({f['score']:.2f})"
            for f in data.get("top_frames", [])[:3]
        )
        frame_table.add_row(source, data.get("primary_frame", "?"), top_frames)

    console.print(frame_table)
    div = frame_comparison.get("frame_divergence_score", 0)
    console.print(f"[green]✓[/green] Frame divergence score: [bold]{div:.3f}[/bold]")

    # ─────────────────────────────────────────────────────────────────
    # PHASE 5: SENTIMENT & TONE ANALYSIS
    # ─────────────────────────────────────────────────────────────────
    print_phase(5, "SENTIMENT & TONE ANALYSIS")

    from src.sentiment.analyzer import SentimentAnalyzer, compare_sentiment

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Analyzing sentiment...", total=None)
        sentiment_analyzer = SentimentAnalyzer()
        cluster_articles = sentiment_analyzer.analyze_batch(cluster_articles)
        sentiment_comparison = compare_sentiment(cluster_articles)
        progress.update(task, completed=True)

    sent_table = Table(title="Sentiment by Source", show_lines=True)
    sent_table.add_column("Source", style="cyan", width=18)
    sent_table.add_column("Compound", style="bold", width=10, justify="right")
    sent_table.add_column("Label", style="white", width=12)
    sent_table.add_column("Dominant Tone", style="dim", width=14)

    for source, data in sentiment_comparison.get("sentiment_by_source", {}).items():
        compound = data.get("compound", 0)
        label = data.get("label", "neutral")
        tone = data.get("tone", "")
        color = "green" if compound > 0.1 else ("red" if compound < -0.1 else "yellow")
        sent_table.add_row(
            source,
            f"[{color}]{compound:+.3f}[/{color}]",
            label.title(),
            tone.title(),
        )

    console.print(sent_table)
    console.print(f"[green]✓[/green] Most negative: [red]{sentiment_comparison.get('most_negative_source')}[/red]")
    console.print(f"[green]✓[/green] Most positive: [green]{sentiment_comparison.get('most_positive_source')}[/green]")
    console.print(f"[green]✓[/green] Sentiment range: [bold]{sentiment_comparison.get('sentiment_range', 0):.3f}[/bold]")

    # ─────────────────────────────────────────────────────────────────
    # PHASE 6: LEXICAL BIAS DETECTION
    # ─────────────────────────────────────────────────────────────────
    print_phase(6, "LEXICAL BIAS DETECTION")

    from src.bias.detector import BiasDetector, compare_bias

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Detecting lexical bias...", total=None)
        bias_detector = BiasDetector()
        cluster_articles = bias_detector.analyze_batch(cluster_articles)
        bias_comparison = compare_bias(cluster_articles)
        progress.update(task, completed=True)

    bias_table = Table(title="Bias Analysis by Source", show_lines=True)
    bias_table.add_column("Source", style="cyan", width=18)
    bias_table.add_column("Bias Score", width=12, justify="right")
    bias_table.add_column("Power Words", style="yellow", width=35)
    bias_table.add_column("Exclusive Vocab", style="dim", width=25)

    excl_vocab = bias_comparison.get("exclusive_vocabulary", {})
    for source, score in bias_comparison.get("bias_scores", {}).items():
        color = "red" if score > 0.5 else ("yellow" if score > 0.25 else "green")
        # Find power words for this source
        power_words = []
        for a in cluster_articles:
            if a["source"] == source:
                power_words = a.get("bias_analysis", {}).get("power_words", [])[:5]
                break
        excl = ", ".join(excl_vocab.get(source, [])[:4])
        bias_table.add_row(
            source,
            f"[{color}]{score:.3f}[/{color}]",
            ", ".join(power_words),
            excl,
        )

    console.print(bias_table)
    console.print(f"[green]✓[/green] Most biased: [red]{bias_comparison.get('most_biased_source')}[/red]")
    console.print(f"[green]✓[/green] Most neutral: [green]{bias_comparison.get('most_neutral_source')}[/green]")

    # ─────────────────────────────────────────────────────────────────
    # PHASE 7: COMPARISON ENGINE
    # ─────────────────────────────────────────────────────────────────
    print_phase(7, "COMPARISON ENGINE — FINAL REPORT")

    from src.comparison.engine import ComparisonEngine

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Running comparison engine...", total=None)
        engine = ComparisonEngine()
        report = engine.compare(cluster_articles)
        progress.update(task, completed=True)

    # ── Final Summary Panel ────────────────────────────────────────
    divergence = report.get("overall_divergence_score", 0)
    div_level = report.get("divergence_level", "Unknown")
    div_color = {
        "Extreme": "red", "High": "yellow", "Moderate": "cyan",
        "Low": "blue", "Minimal": "green"
    }.get(div_level, "white")

    console.print(Panel(
        f"[bold]Event:[/bold] {report.get('event_summary', '')[:150]}\n\n"
        f"[bold]Articles Analyzed:[/bold] {report.get('articles_analyzed')}\n"
        f"[bold]Sources:[/bold] {', '.join(report.get('sources', []))}\n\n"
        f"[bold]Overall Divergence Score:[/bold] [{div_color}]{divergence:.3f} — {div_level}[/{div_color}]",
        title="[bold white]📊 NFIE FINAL REPORT[/bold white]",
        border_style="cyan",
    ))

    console.print("\n[bold]🎯 Key Differences Detected:[/bold]")
    for diff in report.get("key_differences", []):
        label, _, rest = diff.partition(": ")
        console.print(f"  • [bold yellow]{label}:[/bold yellow] {rest}")

    # ── Save full report to JSON ───────────────────────────────────
    report_path = Path("data/processed/full_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove un-serializable numpy arrays
    serializable_report = {k: v for k, v in report.items() if k != "enriched_articles"}
    with open(report_path, "w") as f:
        json.dump(serializable_report, f, indent=2)

    console.print(f"\n[green]✓[/green] Full report saved to [dim]{report_path}[/dim]")
    console.print("\n[bold green]✅ All phases completed successfully![/bold green]")
    console.print("\n[dim]Next steps:[/dim]")
    console.print("  • [cyan]python -m uvicorn src.api.main:app --reload[/cyan]  → Start API server")
    console.print("  • [cyan]streamlit run dashboard/app.py[/cyan]               → Launch dashboard")


if __name__ == "__main__":
    run()
