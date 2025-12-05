from datetime import datetime
from pathlib import Path
from typing import Any, Mapping


def _render_extras(extras: Mapping[str, Any]) -> str:
    if not extras:
        return ""
    html = ['<div class="section"><h2>Debug</h2>']
    for key in ["pipeline", "method", "run_id"]:
        if extras.get(key):
            html.append(f"<p><strong>{key}:</strong> {extras[key]}</p>")
    if extras.get("queries"):
        html.append("<h3>Generated sub-queries</h3><ul>")
        for q in extras["queries"]:
            html.append(f"<li>{q}</li>")
        html.append("</ul>")
    if extras.get("retrieved"):
        html.append("<h3>Retrieved snippets</h3><ol>")
        for r in extras["retrieved"][:20]:
            src = r.get("source", "")
            prev = r.get("preview", "").replace("<", "&lt;").replace(">", "&gt;")
            html.append(f"<li><code>{src}</code> — {prev}</li>")
        html.append("</ol>")
    if extras.get("candidates"):
        html.append(
            (
                "<h3>Rerank candidates</h3>"
                '<table border="1" cellpadding="6" cellspacing="0">'
                "<tr><th>Score</th><th>Source</th><th>Preview</th></tr>"
            )
        )
        for c in extras["candidates"][:30]:
            src = c.get("source", "")
            prev = c.get("preview", "").replace("<", "&lt;").replace(">", "&gt;")
            sc = float(c.get("score", 0.0))
            html.append(f"<tr><td>{sc:.4f}</td><td><code>{src}</code></td><td>{prev}</td></tr>")
        html.append("</table>")
    if extras.get("usage"):
        u = extras["usage"]
        html.append("<h3>Usage</h3><pre>" + str(u) + "</pre>")
    html.append("</div>")
    return "\n".join(html)


def write_simple_report(
    question: str,
    answer: str,
    cfg: Mapping[str, Any],
    extras: Mapping[str, Any] | None = None,
) -> str:
    reports = Path("reports")
    reports.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = reports / f"report-{ts}.html"
    extras_html = _render_extras(extras or {})
    html = (
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>rag-bencher report</title>"
        f"<style>"
        f"body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;"
        f"max-width:900px;margin:2rem auto;padding:0 1rem}}"
        f"code,pre{{background:#f6f8fa;padding:.2rem .3rem;border-radius:6px}}"
        f".section{{margin-bottom:1.5rem}}</style></head>"
        f"<body><h1>rag-bencher report</h1>"
        f"<p><strong>Timestamp:</strong> {ts}</p>"
        f"<div class='section'><h2>Question</h2><p>{question}</p></div>"
        f"<div class='section'><h2>Answer</h2><p>{answer}</p></div>"
        f"{extras_html}"
        f"<div class='section'><h2>Config</h2><pre>{cfg}</pre></div>"
        f"</body></html>"
    )
    path.write_text(html, encoding="utf-8")
    return str(path)
