# T5 News Summarizer (Flask)

[![Live Demo – Hugging Face](https://img.shields.io/badge/Live_Demo-NewsAI%20on%20Hugging%20Face-blue?style=for-the-badge&logo=huggingface)](https://Kabhilan27-NewsAI.hf.space/)
![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Transformers](https://img.shields.io/badge/%F0%9F%A4%97_Transformers-FFCC00?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)
![Status](https://img.shields.io/badge/Build-Stable-success?style=for-the-badge)
![Stars](https://img.shields.io/github/stars/kabhilan27/T5-NewsSummarizer?style=for-the-badge&color=brightgreen)

---

A modern, **dark-themed Flask web app** powered by a **fine-tuned T5 model** for **abstractive news summarization**.  
Paste long articles → get clear, concise summaries — with **style presets**, **export (.txt/.pdf)**, **copy/share**, and a **More Formal Rewrite**.

---

## 🚀 Features

### 🧠 Intelligent Summarization
- Fine-tuned **T5 (Text-to-Text Transfer Transformer)** for abstractive summaries.
- **Style presets:**
  - 🗞️ News (default)
  - 🎓 Academic
  - 💼 Marketing
  - 🗣️ Simple English

### ✨ Modern Web Interface
- **Flask** + custom **HTML/CSS** (glassy, dark-only UI).
- Responsive layout with smooth gradients.
- One-click **Copy**, **Download (.txt/.pdf)**, **Share link**.

### 🧾 Post-Processing
- **“More Formal Rewrite”** for a polished, professional tone (model-generated).
- **PDF export** powered by ReportLab.

### ⚙️ Developer-Friendly
- Model weights tracked with **Git LFS**.
- Clean, extensible structure — ready for deployment or API integration.

---

## 🌐 Deployment

The app is live on **Hugging Face Spaces** using a Docker container setup.  
➡️ **Try it here:** [https://Kabhilan27-NewsAI.hf.space/](https://Kabhilan27-NewsAI.hf.space/)

- **Backend:** Flask served via Gunicorn  
- **Model Host:** Local fine-tuned T5 (or optional Hugging Face Hub model via `MODEL_ID`)  
- **Frontend:** Custom HTML/CSS with a glassmorphic dark theme  

---

## 🧩 Project Structure
```text
T5-NewsSummarizer/
│
├── app.py                   # Flask application (routes + UI)
├── requirements.txt         # Python dependencies
├── Dockerfile               # Deployment configuration (for Hugging Face)
├── .gitignore               # Ignore .venv, .env, and model cache
├── .gitattributes           # Git LFS rules for model files
├── exported_t5_news/        # Fine-tuned T5 (save_pretrained format)
│   ├── config.json
│   ├── generation_config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── spiece.model
└── README.md

```
## 🧠 Model Details

**Architecture:** `T5ForConditionalGeneration`  
- **Base Model:** `t5-base`  
- **Dataset:** *News Summary Dataset* (Kaggle) 
- **Key Libraries:** 🤗 Transformers, PyTorch (Lightning for training), Pandas, NumPy  

---

### 💾 Export (after training)
```python
trained_model.model.save_pretrained("exported_t5_news")
tokenizer.save_pretrained("exported_t5_news")
```

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/kabhilan27/T5-NewsSummarizer.git
cd T5-NewsSummarizer
```

### 2️⃣ Create & Activate Virtual Environment
```bash
python -m venv .venv
Activate the environment:

🪟 Windows
.\.venv\Scripts\activate

🐧 macOS / Linux
source .venv/bin/activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Pull LFS Model Files
```bash
git lfs install
git lfs pull
```

### 5️⃣ Run the Application
```bash
python app.py
Open your browser and visit:
http://127.0.0.1:5000
```

## 🖥️ Usage

1️⃣ **Paste** a news article into the textarea.  
2️⃣ **Pick a Topic/Style preset:**  
   - 🗞️ News  
   - 🎓 Academic  
   - 💼 Marketing  
   - 🗣️ Simple English  
3️⃣ **Click “Summarize.”**  
4️⃣ **Use tools:**  
   - 📋 Copy  
   - 💾 Download `.txt` / `.pdf`  
   - 🔗 Share link  
5️⃣ **Review** the *More Formal Rewrite* displayed beneath the main summary.

## 🧰 Dependencies

| Library        | Purpose                                      |
|----------------|----------------------------------------------|
| **flask**      | Web server & routing                         |
| **torch**      | PyTorch backend                              |
| **transformers** | Hugging Face T5 model + tokenizer           |
| **reportlab**  | PDF export                                   |

> *(Pandas, NumPy, and PyTorch Lightning were used during training and are **not required** to run the app.)*

## 🧑‍💻 Author

**Kabhilan Chandrasekaran**  
🔗 **GitHub:** [github.com/kabhilan27](https://github.com/kabhilan27)

## 🪪 License

**MIT License** — feel free to use, modify, and share with proper attribution.

## 🌟 Support

If you find this project useful, please ⭐ **star the repository** and share it!  
Pull requests for **UI improvements**, **new features**, or **deployment recipes** are always welcome.
