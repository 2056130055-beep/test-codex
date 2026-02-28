from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from pair_trading.config import Config
from pair_trading.dashboard.charts import prepare_heatmap_data


def build_dashboard(results: list[dict], corr_matrix, output_html: str) -> None:
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    tpl = env.get_template("dashboard.html")

    top_results = results[: Config.TOP_PAIRS_DISPLAY]
    stats = {
        "total": len(results),
        "buy": sum(1 for r in results if r["signal"] == "BUY"),
        "sell": sum(1 for r in results if r["signal"] == "SELL"),
        "watch": sum(1 for r in results if str(r["signal"]).startswith("WATCH")),
        "coint": sum(1 for r in results if r["pval"] <= 0.10),
        "stable": sum(
            1
            for r in results
            if r["recent_coint_stable"] and r["corr_stability"] > Config.CORR_STABILITY_THRESHOLD
        ),
    }

    html = tpl.render(
        results=top_results,
        results_json=json.dumps(top_results),
        heatmap_json=json.dumps(prepare_heatmap_data(corr_matrix, top_n=20)),
        stats=stats,
    )
    Path(output_html).write_text(html, encoding="utf-8")
