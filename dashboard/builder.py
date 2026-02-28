from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from config import Config


def build_dashboard(results: list[dict], corr_matrix, output_html: str):
    tickers = corr_matrix.columns[:20].tolist()
    heatmap = corr_matrix.loc[tickers, tickers].round(3).values.tolist() if len(tickers) else []

    summary = {
        "total": len(results),
        "buy": sum(1 for r in results if r["signal"] == "BUY"),
        "sell": sum(1 for r in results if r["signal"] == "SELL"),
        "watch": sum(1 for r in results if r["signal"].startswith("WATCH")),
        "coint": sum(1 for r in results if r["pval"] <= 0.10),
    }
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(template_dir))
    tpl = env.get_template("dashboard.html")
    html = tpl.render(
        config={"corr_stability": Config.CORR_STABILITY_THRESHOLD},
        summary=summary,
        top_pair=results[0] if results else None,
        pairs_json=json.dumps(results[: Config.TOP_PAIRS_DISPLAY]),
        heatmap_json=json.dumps({"tickers": tickers, "values": heatmap}),
    )
    Path(output_html).write_text(html, encoding="utf-8")
