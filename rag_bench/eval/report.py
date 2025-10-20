
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

def _render_extras(extras: Dict[str, Any]) -> str:
    if not extras:
        return ""
    html = ['<div class="section"><h2>Debug</h2>']
    if extras.get("pipeline"):
        html.append(f"<p><strong>Pipeline:</strong> {extras['pipeline']}</p>")
    if extras.get("method"):
        html.append(f"<p><strong>Method:</strong> {extras['method']}</p>")
    if extras.get("queries"):
        html.append("<h3>Generated sub-queries</h3><ul>")
        for q in extras["queries"]:
            html.append(f"<li>{q}</li>")
        html.append("</ul>")
    if extras.get("retrieved"):
        html.append("<h3>Retrieved snippets</h3><ol>")
        for r in extras["retrieved"][:20]:
            src = r.get("source","")
            prev = r.get("preview","").replace("<","&lt;").replace(">","&gt;")
            html.append(f"<li><code>{src}</code> — {prev}</li>")
        html.append("</ol>")
    if extras.get("candidates"):
        html.append("<h3>Rerank candidates</h3>")
        html.append('<table border="1" cellpadding="6" cellspacing="0"><tr><th>Score</th><th>Source</th><th>Preview</th></tr>')
        for c in extras["candidates"][:30]:
            src = c.get("source","")
            prev = c.get("preview","").replace("<","&lt;").replace(">","&gt;")
            score = c.get("score", 0.0)
            html.append(f"<tr><td>{score:.4f}</td><td><code>{src}</code></td><td>{prev}</td></tr>")
        html.append("</table>")
    html.append("</div>")
    return "\n".join(html)

def write_simple_report(question: str, answer: str, cfg: Dict, extras: Dict | None = None) -> str:
    reports = Path("reports")
    reports.mkdir(exist_ok=True, parents=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = reports / f"report-{ts}.html"
    extras_html = _render_extras(extras or {})
    html = f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>rag-bench report</title>
<style>
body{{font-family:system-ui, -apple-system, Segoe UI, Roboto, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem;}}
code, pre{{background:#f6f8fa; padding: .2rem .3rem; border-radius: 6px;}}
.section{{margin-bottom: 1.5rem;}}
table{{border-collapse: collapse; width: 100%;}}
th, td{{text-align: left;}}
</style></head>
<body>
<h1>rag-bench report</h1>
<p><strong>Timestamp:</strong> {ts}</p>
<div class="section"><h2>Question</h2><p>{question}</p></div>
<div class="section"><h2>Answer</h2><p>{answer}</p></div>
{extras_html}
<div class="section"><h2>Config</h2><pre>{cfg}</pre></div>
</body></html>
"""
    path.write_text(html, encoding="utf-8")
    return str(path)
