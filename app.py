# app.py — T5 News Summarizer (mobile-responsive, centered summarize + clean buttons)
import os, io, textwrap
import torch
from flask import Flask, request, jsonify, render_template_string, send_file
from transformers import T5ForConditionalGeneration, T5TokenizerFast as T5Tokenizer

# ---------- Config ----------
MODEL_DIR = os.getenv("MODEL_DIR", "exported_t5_news")
MAX_INPUT_LEN = int(os.getenv("MAX_INPUT_LEN", 512))
MAX_SUM_LEN = int(os.getenv("MAX_SUM_LEN", 150))
NUM_BEAMS = int(os.getenv("NUM_BEAMS", 2))
LENGTH_PENALTY = float(os.getenv("LENGTH_PENALTY", 1.0))
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", 2.5))

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Load model ----------
tokenizer = T5Tokenizer.from_pretrained(MODEL_DIR)
model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR).to(device)
model.eval()


def _generate(prompt: str,
              max_len: int = MAX_SUM_LEN,
              num_beams: int = NUM_BEAMS,
              length_penalty: float = LENGTH_PENALTY,
              repetition_penalty: float = REPETITION_PENALTY) -> str:
    enc = tokenizer(
        prompt, max_length=MAX_INPUT_LEN, padding="max_length",
        truncation=True, return_tensors="pt",
    )
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)
    with torch.no_grad():
        gen_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attn,
            max_length=max_len,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            early_stopping=True,
        )
    return tokenizer.decode(gen_ids[0], skip_special_tokens=True)


def apply_style(text: str, style: str) -> str:
    s = (style or "news").lower()
    if s == "academic":
        return f"Summarize formally and objectively with precise wording:\n{text}"
    if s == "marketing":
        return f"Summarize persuasively highlighting key benefits:\n{text}"
    if s == "simple-english":
        return f"Summarize in simple English for a Grade 9 reader:\n{text}"
    return f"Summarize clearly and concisely:\n{text}"


def summarize(src_text: str, style: str) -> str:
    return _generate(apply_style(src_text, style))


def rewrite_formal(summary: str) -> str:
    return _generate(
        f"Rewrite this summary in a formal, neutral, and polished tone:\n{summary}",
        max_len=MAX_SUM_LEN,
        num_beams=4,
        length_penalty=1.0,
        repetition_penalty=2.6,
    )


# ---------- Flask ----------
app = Flask(__name__)

INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0, viewport-fit=cover" />
<title>T5 News Summarizer</title>
<link rel="icon" href="data:,">
<meta name="theme-color" content="#0b0f17">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">

<style>
  :root{
    --bg:#0b0f17; --bg2:#0a0d14;
    --panel:rgba(12,16,24,.55);
    --card:rgba(255,255,255,.06);
    --border:rgba(255,255,255,.12);
    --text:#e8eef9; --muted:#a7b0c0;
    --accent:#5cf0d0; --accent-2:#6ea8ff;
  }
  *{box-sizing:border-box;margin:0;padding:0}
  body{
    font-family:Inter,system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
    background:
      radial-gradient(1200px 800px at var(--mx,50%) var(--my,30%), rgba(92,240,208,.12), transparent 60%),
      radial-gradient(900px 600px at calc(100% - 10%) 10%, rgba(110,168,255,.10), transparent 60%),
      linear-gradient(180deg,var(--bg) 0%,var(--bg2) 100%);
    color:var(--text); line-height:1.5;
  }

  .wrap{min-height:100vh; display:flex; justify-content:center; align-items:flex-start; padding:16px}
  .card{
    width:100%; max-width:680px;
    background:var(--card);
    border:1px solid var(--border);
    border-radius:20px;
    box-shadow:0 10px 40px rgba(0,0,0,.35);
    backdrop-filter: blur(16px) saturate(140%);
  }

  .header{padding:16px; border-bottom:1px solid var(--border); display:flex; align-items:center; gap:12px;}
  .logo{
    width:36px;height:36px;border-radius:10px;
    background:linear-gradient(135deg,var(--accent),var(--accent-2));
    display:grid;place-items:center;font-weight:800;color:#08121a;
  }

  h1{font-size:20px;margin:0;}
  .muted{color:var(--muted);font-size:13px}

  .content{display:grid;gap:12px;padding:16px}

  textarea, select{
    width:100%;background:var(--panel);color:var(--text);
    border:1px solid var(--border);border-radius:14px;padding:12px 14px;
  }
  textarea{min-height:180px;resize:vertical;}
  textarea:focus, select:focus{border-color:var(--accent);outline:none;box-shadow:0 0 0 4px rgba(92,240,208,.15);}
  select option{background:#101821;color:#e8eef9;}

  .btn{
    background:linear-gradient(120deg,var(--accent),var(--accent-2));
    color:#071218;border:none;border-radius:14px;padding:12px 20px;
    font-weight:700;cursor:pointer;transition:0.2s ease;
  }
  .btn:hover{filter:brightness(1.05);} .btn:active{transform:translateY(1px);}

  .center-row{display:flex;justify-content:center;align-items:center;margin-top:10px;}

  /* Horizontal button bar */
  .summary-header{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;}
  .action-bar{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:8px;}
  .chip-btn{
    border:1px solid var(--border);border-radius:10px;background:linear-gradient(135deg,var(--accent),var(--accent-2));
    color:#071218;font-weight:600;padding:8px 14px;cursor:pointer;
    transition:all 0.2s ease;box-shadow:0 4px 12px rgba(110,168,255,0.2);
  }
  .chip-btn:hover{transform:translateY(-1px);filter:brightness(1.1);}
  .summary{
    white-space:pre-wrap;background:var(--panel);border:1px solid var(--border);
    border-radius:14px;padding:16px;margin-top:12px;
  }

  .formal-card{background:linear-gradient(180deg,rgba(255,255,255,.04),rgba(255,255,255,.02));
    border:1px solid var(--border);border-radius:16px;padding:16px 18px;margin-top:16px;}
</style>
</head>

<body>
<div class="wrap">
  <div class="card">
    <div class="header">
      <div class="logo">T5</div>
      <div>
        <h1>T5 News Summarizer</h1>
        <div class="muted">Copy / Download / Share · Style presets · Formal variant</div>
      </div>
    </div>

    <form class="content" method="post" onsubmit="startLoading()">
      <label class="muted">Topic / Style</label>
      <select name="style">
        <option value="news" {% if style=='news' %}selected{% endif %}>News (default)</option>
        <option value="academic" {% if style=='academic' %}selected{% endif %}>Academic</option>
        <option value="marketing" {% if style=='marketing' %}selected{% endif %}>Marketing</option>
        <option value="simple-english" {% if style=='simple-english' %}selected{% endif %}>Simple English</option>
      </select>

      <label class="muted" for="text">Paste article here…</label>
      <textarea id="text" name="text" placeholder="Paste article here..." required>{{text}}</textarea>

      <div class="center-row">
        <button class="btn" id="btnSubmit" type="submit">Summarize</button>
      </div>
    </form>

    {% if summary %}
    <div class="content">
      <div class="summary-header">
        <div class="muted">Summary</div>
        <div class="action-bar">
          <button class="chip-btn" onclick="copyOut()">Copy</button>
          <button class="chip-btn" onclick="downloadTxt()">.txt</button>
          <button class="chip-btn" onclick="downloadPdf()">.pdf</button>
          <button class="chip-btn" onclick="shareLink()">Share</button>
        </div>
      </div>
      <div id="out" class="summary">{{summary}}</div>
      <div class="formal-card">
        <h3 class="muted">More formal rewrite</h3>
        <div id="v_formal" class="summary">{{v_formal}}</div>
      </div>
    </div>
    {% endif %}
  </div>
</div>

<script>
function startLoading(){
  const b=document.getElementById('btnSubmit');
  b.disabled=true;setTimeout(()=>b.disabled=false,1500);
}
function copyOut(){
  const el=document.getElementById('out');if(el)navigator.clipboard.writeText(el.innerText);
}
function downloadTxt(){
  const el=document.getElementById('out');if(!el)return;
  const blob=new Blob([el.innerText],{type:'text/plain'});
  const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='summary.txt';a.click();
}
async function downloadPdf(){
  const el=document.getElementById('out');if(!el)return alert('No summary yet');
  const t=el.innerText.trim();if(!t)return alert('No summary yet');
  const res=await fetch('/export/pdf',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:t})});
  if(!res.ok){alert('PDF export requires reportlab');return;}
  const blob=await res.blob();const a=document.createElement('a');
  a.href=URL.createObjectURL(blob);a.download='summary.pdf';a.click();
}
function shareLink(){
  const t=(document.getElementById('text')?.value||'').trim();
  if(!t)return alert('Paste an article first');
  const s=encodeURIComponent(btoa(unescape(encodeURIComponent(t))));
  history.replaceState(null,'',location.pathname+'#q='+s);
  alert('Shareable link added to the URL.');
}
</script>
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    style = request.form.get("style", "news")
    text, summary, v_formal = "", "", ""
    if request.method == "POST":
        text = (request.form.get("text") or "").strip()
        if text:
            summary = summarize(text, style)
            v_formal = rewrite_formal(summary)
    return render_template_string(INDEX_HTML, text=text, summary=summary, style=style, v_formal=v_formal)


@app.post("/export/pdf")
def export_pdf():
    data = request.get_json(force=True, silent=True) or {}
    content = (data.get("text") or "").strip()
    if not content:
        return "No content", 400
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.pdfgen import canvas
    except Exception:
        return "reportlab not installed", 400
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER
    y = height - 72
    c.setFont("Helvetica-Bold", 14)
    c.drawString(72, y, "Summary")
    y -= 24
    c.setFont("Helvetica", 11)
    for line in textwrap.wrap(content, 95):
        if y < 72:
            c.showPage(); y = height - 72; c.setFont("Helvetica", 11)
        c.drawString(72, y, line); y -= 14
    c.showPage(); c.save(); buffer.seek(0)
    return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name="summary.pdf")

@app.get("/health")
def health():
    return {"ok": True}

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", 5000)), debug=True)
