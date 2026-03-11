import csv
import json
from pathlib import Path


def read_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def as_int(value, default=0):
    try:
        return int(float(value))
    except Exception:
        return default


def main():
    repo_root = Path(__file__).resolve().parent.parent
    logs_root = repo_root / "tuning_logs"
    out_dir = logs_root / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(logs_root.rglob("tuning_log_tl*_summary_desc.csv"))

    by_model = {}
    top_rows = []
    rank_curves = {}

    for path in csv_files:
        stem_lower = path.stem.lower()
        if stem_lower.endswith("tl1_summary_desc"):
            arch = "tl1"
        elif stem_lower.endswith("tl2_summary_desc"):
            arch = "tl2"
        else:
            continue

        model = path.parent.name
        rows = read_rows(path)
        if not rows:
            continue

        sorted_rows = sorted(rows, key=lambda r: as_float(r.get("tokens_per_second")), reverse=True)
        best = sorted_rows[0]

        by_model.setdefault(model, {})[arch] = as_float(best.get("tokens_per_second"))

        top_rows.append(
            {
                "model": model,
                "arch": arch,
                "tokens_per_second": as_float(best.get("tokens_per_second")),
                "M": as_int(best.get("M")),
                "K": as_int(best.get("K")),
                "BM": as_int(best.get("BM")),
                "BK": as_int(best.get("BK")),
                "bm": as_int(best.get("bm")),
                "avg_latency_s": as_float(best.get("avg_latency_s")),
                "source": str(path.relative_to(repo_root)).replace("\\\\", "/"),
            }
        )

        top10 = sorted_rows[:10]
        rank_curves[f"{model}-{arch.upper()}"] = [
            as_float(r.get("tokens_per_second")) for r in top10
        ]

    model_order = sorted(by_model.keys(), key=lambda m: (
        len(m), m.lower()
    ))
    tl1_values = [by_model.get(m, {}).get("tl1", 0.0) for m in model_order]
    tl2_values = [by_model.get(m, {}).get("tl2", 0.0) for m in model_order]

    top_rows = sorted(top_rows, key=lambda x: x["tokens_per_second"], reverse=True)

    payload = {
        "models": model_order,
        "tl1": tl1_values,
        "tl2": tl2_values,
        "top_rows": top_rows,
        "rank_curves": rank_curves,
    }

    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
  <title>BitNet Tuning Visualization</title>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js\"></script>
  <style>
    :root {{
      --bg: #f4f1ea;
      --panel: #fffdf8;
      --ink: #232323;
      --muted: #6b6b6b;
      --accent-a: #1f6f8b;
      --accent-b: #d1495b;
      --line: #e3dbcf;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 10% 10%, #efe6d6 0%, transparent 45%),
        radial-gradient(circle at 90% 0%, #e3efe8 0%, transparent 35%),
        var(--bg);
    }}
    .wrap {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: 28px; }}
    p {{ margin: 0 0 18px; color: var(--muted); }}
    .grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 16px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 8px 24px rgba(0,0,0,0.04);
    }}
    .card h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px; text-align: left; white-space: nowrap; }}
    th {{ background: #f8f4ec; }}
    .footer {{ margin-top: 14px; font-size: 12px; color: var(--muted); }}
    @media (min-width: 900px) {{
      .grid {{ grid-template-columns: 1fr 1fr; }}
      .span-2 {{ grid-column: span 2; }}
    }}
  </style>
</head>
<body>
  <div class=\"wrap\">
    <h1>BitNet Kernel Tuning Dashboard</h1>
    <p>Generated from <code>tuning_log_tl*_summary_desc.csv</code> files.</p>

    <div class=\"grid\">
      <section class=\"card\">
        <h2>Top Throughput by Model</h2>
        <canvas id=\"barTop\"></canvas>
      </section>

      <section class=\"card\">
        <h2>Top-10 Candidate Curves</h2>
        <canvas id=\"lineTop10\"></canvas>
      </section>

      <section class=\"card span-2\">
        <h2>Best Configs (All Model/Arch)</h2>
        <div class=\"table-wrap\">
          <table>
            <thead>
              <tr>
                <th>Model</th><th>Arch</th><th>TPS</th><th>M</th><th>K</th><th>BM</th><th>BK</th><th>bm</th><th>avg_latency_s</th><th>source</th>
              </tr>
            </thead>
            <tbody id=\"bestTable\"></tbody>
          </table>
        </div>
      </section>
    </div>

    <div class=\"footer\">Output: tuning_logs/visualization/tuning_dashboard.html</div>
  </div>

  <script>
    const data = {json.dumps(payload)};

    const curveLabels = Array.from({{length: 10}}, (_, i) => `rank ${{i + 1}}`);
    const curveDatasets = Object.entries(data.rank_curves).map(([name, vals], i) => {{
      const hue = (i * 29) % 360;
      return {{
        label: name,
        data: vals,
        borderColor: `hsl(${{hue}}, 60%, 45%)`,
        backgroundColor: `hsla(${{hue}}, 60%, 45%, 0.15)`,
        tension: 0.25,
        fill: false
      }};
    }});

    if (typeof Chart !== 'undefined') {{
      new Chart(document.getElementById('barTop'), {{
        type: 'bar',
        data: {{
          labels: data.models,
          datasets: [
            {{ label: 'TL1', data: data.tl1, backgroundColor: 'rgba(31,111,139,0.75)' }},
            {{ label: 'TL2', data: data.tl2, backgroundColor: 'rgba(209,73,91,0.75)' }}
          ]
        }},
        options: {{
          responsive: true,
          plugins: {{ legend: {{ position: 'top' }} }},
          scales: {{ y: {{ title: {{ display: true, text: 'tokens/s' }} }} }}
        }}
      }});

      new Chart(document.getElementById('lineTop10'), {{
        type: 'line',
        data: {{ labels: curveLabels, datasets: curveDatasets }},
        options: {{
          responsive: true,
          plugins: {{ legend: {{ position: 'bottom' }} }},
          scales: {{ y: {{ title: {{ display: true, text: 'tokens/s' }} }} }}
        }}
      }});
    }} else {{
      console.warn('Chart.js failed to load. Rendering table only.');
    }}

    const tbody = document.getElementById('bestTable');
    data.top_rows.forEach(r => {{
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td>${{r.model}}</td>
        <td>${{r.arch.toUpperCase()}}</td>
        <td>${{r.tokens_per_second.toFixed(3)}}</td>
        <td>${{r.M}}</td>
        <td>${{r.K}}</td>
        <td>${{r.BM}}</td>
        <td>${{r.BK}}</td>
        <td>${{r.bm}}</td>
        <td>${{r.avg_latency_s.toExponential(3)}}</td>
        <td><code>${{r.source}}</code></td>
      `;
      tbody.appendChild(tr);
    }});
  </script>
</body>
</html>
"""

    out_html = out_dir / "tuning_dashboard.html"
    out_html.write_text(html, encoding="utf-8")

    print(f"Wrote dashboard: {out_html}")


if __name__ == "__main__":
    main()
