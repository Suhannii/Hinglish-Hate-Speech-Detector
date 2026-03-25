"""
app.py
------
Flask web application for Hinglish Hate Speech Detection.
Run with: python app.py
Then open: http://localhost:5000
"""

import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, request, jsonify, render_template_string
from src.predict import HateSpeechPredictor

app = Flask(__name__)

# ── Load model once at startup ─────────────────────────────────────────────────
MODEL_DIR  = "models/muril_model"
BASE_MODEL = "google/muril-base-cased"

# Fallback to mBERT if MuRIL not trained yet
if not os.path.exists(MODEL_DIR):
    MODEL_DIR  = "models/saved_model"
    BASE_MODEL = "bert-base-multilingual-cased"

print(f"[app] Loading model from: {MODEL_DIR}")
predictor = HateSpeechPredictor(model_dir=MODEL_DIR, base_model=BASE_MODEL)

# ── HTML template (single-file, no external files needed) ─────────────────────
HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Hinglish Hate Speech Detector</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      font-family: 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 20px;
    }

    .card {
      background: rgba(255,255,255,0.05);
      backdrop-filter: blur(12px);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 20px;
      padding: 40px;
      width: 100%;
      max-width: 680px;
      color: #fff;
      box-shadow: 0 25px 50px rgba(0,0,0,0.4);
    }

    .badge {
      display: inline-block;
      background: linear-gradient(90deg, #f093fb, #f5576c);
      color: white;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 1.5px;
      text-transform: uppercase;
      padding: 4px 12px;
      border-radius: 20px;
      margin-bottom: 16px;
    }

    h1 {
      font-size: 26px;
      font-weight: 700;
      margin-bottom: 6px;
      background: linear-gradient(90deg, #a18cd1, #fbc2eb);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }

    .subtitle {
      color: rgba(255,255,255,0.5);
      font-size: 13px;
      margin-bottom: 28px;
    }

    textarea {
      width: 100%;
      background: rgba(255,255,255,0.08);
      border: 1px solid rgba(255,255,255,0.15);
      border-radius: 12px;
      color: #fff;
      font-size: 15px;
      padding: 16px;
      resize: vertical;
      min-height: 110px;
      outline: none;
      transition: border 0.2s;
    }
    textarea::placeholder { color: rgba(255,255,255,0.3); }
    textarea:focus { border-color: #a18cd1; }

    .btn {
      margin-top: 14px;
      width: 100%;
      padding: 14px;
      border: none;
      border-radius: 12px;
      background: linear-gradient(90deg, #a18cd1, #fbc2eb);
      color: #1a1a2e;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      transition: opacity 0.2s, transform 0.1s;
    }
    .btn:hover  { opacity: 0.9; }
    .btn:active { transform: scale(0.98); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }

    #result {
      margin-top: 24px;
      display: none;
    }

    .result-box {
      border-radius: 14px;
      padding: 20px 24px;
      border: 1px solid rgba(255,255,255,0.1);
    }
    .result-hate     { background: rgba(245, 87, 108, 0.15); border-color: #f5576c; }
    .result-non-hate { background: rgba(67, 233, 123, 0.12); border-color: #43e97b; }

    .result-label {
      font-size: 22px;
      font-weight: 800;
      margin-bottom: 4px;
    }
    .hate-text     { color: #f5576c; }
    .non-hate-text { color: #43e97b; }

    .result-conf {
      font-size: 13px;
      color: rgba(255,255,255,0.6);
      margin-bottom: 16px;
    }

    .bar-wrap {
      margin-bottom: 10px;
    }
    .bar-label {
      display: flex;
      justify-content: space-between;
      font-size: 12px;
      color: rgba(255,255,255,0.6);
      margin-bottom: 4px;
    }
    .bar-bg {
      background: rgba(255,255,255,0.1);
      border-radius: 6px;
      height: 8px;
      overflow: hidden;
    }
    .bar-fill {
      height: 100%;
      border-radius: 6px;
      transition: width 0.6s ease;
    }
    .bar-hate     { background: linear-gradient(90deg, #f5576c, #f093fb); }
    .bar-non-hate { background: linear-gradient(90deg, #43e97b, #38f9d7); }

    .examples {
      margin-top: 28px;
      border-top: 1px solid rgba(255,255,255,0.08);
      padding-top: 20px;
    }
    .examples h3 {
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 1px;
      color: rgba(255,255,255,0.4);
      margin-bottom: 10px;
    }
    .example-chip {
      display: inline-block;
      background: rgba(255,255,255,0.07);
      border: 1px solid rgba(255,255,255,0.1);
      border-radius: 20px;
      padding: 6px 14px;
      font-size: 12px;
      color: rgba(255,255,255,0.7);
      cursor: pointer;
      margin: 4px 4px 4px 0;
      transition: background 0.2s;
    }
    .example-chip:hover { background: rgba(255,255,255,0.14); }

    .spinner {
      display: inline-block;
      width: 16px; height: 16px;
      border: 2px solid rgba(0,0,0,0.3);
      border-top-color: #1a1a2e;
      border-radius: 50%;
      animation: spin 0.7s linear infinite;
      vertical-align: middle;
      margin-right: 6px;
    }
    @keyframes spin { to { transform: rotate(360deg); } }

    .model-tag {
      margin-top: 20px;
      text-align: center;
      font-size: 11px;
      color: rgba(255,255,255,0.25);
    }
  </style>
</head>
<body>
<div class="card">
  <div class="badge">NLP Research Project</div>
  <h1>Hinglish Hate Speech Detector</h1>
  <p class="subtitle">Powered by MuRIL — Google's multilingual model for Indian languages</p>

  <textarea id="inputText" placeholder="Type a Hinglish sentence here...
e.g. Yeh log bahut gande hain, inhe yahan se nikalo"></textarea>

  <button class="btn" id="analyzeBtn" onclick="analyze()">Analyze Text</button>

  <div id="result">
    <div class="result-box" id="resultBox">
      <div class="result-label" id="resultLabel"></div>
      <div class="result-conf" id="resultConf"></div>
      <div class="bar-wrap">
        <div class="bar-label"><span>Non-Hate</span><span id="nonHatePct"></span></div>
        <div class="bar-bg"><div class="bar-fill bar-non-hate" id="nonHateBar" style="width:0%"></div></div>
      </div>
      <div class="bar-wrap">
        <div class="bar-label"><span>Hate</span><span id="hatePct"></span></div>
        <div class="bar-bg"><div class="bar-fill bar-hate" id="hateBar" style="width:0%"></div></div>
      </div>
    </div>
  </div>

  <div class="examples">
    <h3>Try an example</h3>
    <span class="example-chip" onclick="setExample(this)">Yeh log bahut gande hain</span>
    <span class="example-chip" onclick="setExample(this)">Aaj mausam bahut achha hai</span>
    <span class="example-chip" onclick="setExample(this)">Tum log kisi kaam ke nahi ho</span>
    <span class="example-chip" onclick="setExample(this)">Mujhe chai bahut pasand hai</span>
    <span class="example-chip" onclick="setExample(this)">Is community ke log sab chor hote hain</span>
    <span class="example-chip" onclick="setExample(this)">Yeh movie bahut achi thi</span>
  </div>

  <div class="model-tag">Model: {{ model_name }} &nbsp;|&nbsp; B.Tech NLP Project 2025</div>
</div>

<script>
  function setExample(el) {
    document.getElementById('inputText').value = el.textContent;
    analyze();
  }

  async function analyze() {
    const text = document.getElementById('inputText').value.trim();
    if (!text) return;

    const btn = document.getElementById('analyzeBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span>Analyzing...';

    try {
      const res  = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text })
      });
      const data = await res.json();

      const isHate = data.label_id === 1;
      const box    = document.getElementById('resultBox');
      box.className = 'result-box ' + (isHate ? 'result-hate' : 'result-non-hate');

      document.getElementById('resultLabel').innerHTML =
        `<span class="${isHate ? 'hate-text' : 'non-hate-text'}">${isHate ? '⚠️ Hate Speech' : '✅ Non-Hate'}</span>`;
      document.getElementById('resultConf').textContent =
        `Confidence: ${data.confidence}%  |  Cleaned: "${data.cleaned_text}"`;

      const nh = data.probabilities['Non-Hate'];
      const h  = data.probabilities['Hate'];
      document.getElementById('nonHatePct').textContent = nh + '%';
      document.getElementById('hatePct').textContent    = h  + '%';
      document.getElementById('nonHateBar').style.width = nh + '%';
      document.getElementById('hateBar').style.width    = h  + '%';

      document.getElementById('result').style.display = 'block';
    } catch(e) {
      alert('Error: ' + e.message);
    } finally {
      btn.disabled = false;
      btn.innerHTML = 'Analyze Text';
    }
  }

  document.getElementById('inputText').addEventListener('keydown', e => {
    if (e.key === 'Enter' && e.ctrlKey) analyze();
  });
</script>
</body>
</html>
"""

# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    model_display = BASE_MODEL.split("/")[-1]
    return render_template_string(HTML, model_name=model_display)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400
    result = predictor.predict(text)
    return jsonify(result)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": BASE_MODEL})


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  Hinglish Hate Speech Detector — Web UI")
    print("  Open: http://localhost:5000")
    print("=" * 50 + "\n")
    app.run(debug=False, host="0.0.0.0", port=5000)
