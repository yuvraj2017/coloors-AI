# ============================================================
# Color Intelligence Platform — Enterprise Streamlit Demo
# Single-file, production-grade, free LLM, batch CSV support
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import colorsys
import json
import re
from io import BytesIO
from sklearn.metrics.pairwise import euclidean_distances
from transformers import pipeline
from functools import lru_cache

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Color Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎨 Color Intelligence Platform")
st.caption("Enterprise AI for Color Psychology, Branding & Accessibility")

# ============================================================
# LOAD FREE LLM (LOCAL, NO API)
# ============================================================
@st.cache_resource
def load_llm():
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_length=1024,
        temperature=0.2
    )

llm = load_llm()

# ============================================================
# MASTER PROMPT (CRITICAL)
# ============================================================
MASTER_PROMPT = """
You are a professional color theorist, brand strategist, and design historian.

Your task is to generate a structured, premium-quality color profile
similar in tone, depth, and clarity to Coolors.co.

STRICT RULES:
- Use elegant, professional language
- No emojis
- No marketing fluff
- No repetition
- No speculation beyond color theory
- Output MUST follow the exact section structure below
- Keep descriptions vivid but concise
- Maintain consistency across all colors

SECTIONS TO GENERATE (IN ORDER):

Description:
A refined sensory description of the color.

Psychology:
Explain emotional and psychological responses.

Meaning:
Symbolic and cultural interpretations.

Why use this color:
Practical design and branding use cases.

History:
Historical or stylistic references.

Accessibility:
Interpret accessibility notes clearly (do NOT invent metrics).

Similar colors:
List 3 similar colors with HEX codes and brief distinctions.

COLOR INPUT DATA:
{color_data}

BEGIN OUTPUT:
"""

# ============================================================
# COLOR UTILITIES
# ============================================================
HEX_REGEX = re.compile(r"^#?[0-9A-Fa-f]{6}$")

def normalize_hex(hex_color: str) -> str:
    hex_color = hex_color.strip()
    if not HEX_REGEX.match(hex_color):
        raise ValueError(f"Invalid HEX color: {hex_color}")
    if not hex_color.startswith("#"):
        hex_color = "#" + hex_color
    return hex_color.upper()

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip("#")
    return np.array([
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16)
    ])

def extract_features(hex_color):
    r, g, b = hex_to_rgb(hex_color) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    return {
        "hex": hex_color,
        "hue_deg": round(h * 360, 2),
        "lightness_pct": round(l * 100, 2),
        "saturation_pct": round(s * 100, 2),
        "temperature": "Warm" if (h < 0.15 or h > 0.85) else "Cool"
    }

def accessibility_analysis(features):
    l = features["lightness_pct"]
    if l < 30:
        return "Very dark tone. Best for accents. Avoid long text usage."
    elif l < 45:
        return "Dark tone. Suitable for headings and emphasis."
    elif l < 70:
        return "Balanced tone. Works well for UI elements and highlights."
    else:
        return "Light tone. Use on dark backgrounds for best contrast."

# ============================================================
# SIMILAR COLOR ENGINE (DETERMINISTIC)
# ============================================================
def compute_similar_colors(target_hex, all_hexes, top_n=3):
    target_rgb = hex_to_rgb(target_hex).reshape(1, -1)
    palette_rgb = np.array([hex_to_rgb(h) for h in all_hexes])

    distances = euclidean_distances(target_rgb, palette_rgb)[0]
    sorted_idx = np.argsort(distances)

    similar = []
    for idx in sorted_idx[1:top_n + 1]:
        similar.append(all_hexes[idx])

    return similar

# ============================================================
# LLM GENERATION (CACHED)
# ============================================================
@lru_cache(maxsize=256)
def generate_color_profile(color_payload: str) -> str:
    prompt = MASTER_PROMPT.format(color_data=color_payload)
    output = llm(prompt)[0]["generated_text"]
    return output.strip()

# ============================================================
# FILE EXPORT UTILITIES
# ============================================================
def generate_markdown(results):
    md = "# Color Intelligence Report\n\n"
    for r in results:
        md += f"## {r['name'] or r['hex']}\n"
        md += f"**HEX:** {r['hex']}\n\n"
        md += r["ai_output"] + "\n\n---\n\n"
    return md

def generate_json(results):
    return json.dumps(results, indent=2)

def to_downloadable_file(content, filename, mime):
    buffer = BytesIO()
    buffer.write(content.encode("utf-8"))
    buffer.seek(0)
    return buffer, filename, mime

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.header("Configuration")
st.sidebar.markdown("**LLM:** FLAN-T5 Large (Free, Local)")
st.sidebar.markdown("**Mode:** Enterprise Demo")
st.sidebar.markdown("**Output:** Markdown + JSON")

# ============================================================
# CSV UPLOAD
# ============================================================
uploaded_file = st.file_uploader("Upload CSV with HEX colors", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    if not {"hex"}.issubset(df.columns):
        st.error("CSV must contain a 'hex' column")
        st.stop()

    df["hex"] = df["hex"].apply(normalize_hex)
    df["name"] = df["name"] if "name" in df.columns else None

    st.success(f"Loaded {len(df)} colors")

    all_hexes = df["hex"].tolist()
    results = []

    for _, row in df.iterrows():
        hex_color = row["hex"]
        name = row.get("name")

        features = extract_features(hex_color)
        accessibility = accessibility_analysis(features)
        similar = compute_similar_colors(hex_color, all_hexes)

        color_payload = {
            "name": name,
            "hex": hex_color,
            "features": features,
            "accessibility": accessibility,
            "similar_colors": similar
        }

        ai_output = generate_color_profile(json.dumps(color_payload, indent=2))

        results.append({
            "name": name,
            "hex": hex_color,
            "features": features,
            "accessibility": accessibility,
            "similar_colors": similar,
            "ai_output": ai_output
        })

        # UI RENDER
        st.markdown("---")
        c1, c2 = st.columns([1, 3])

        with c1:
            st.markdown(
                f"""
                <div style="height:160px;
                background:{hex_color};
                border-radius:12px;
                border:1px solid #ddd;"></div>
                """,
                unsafe_allow_html=True
            )
            st.code(hex_color)
            st.json(features)

        with c2:
            st.markdown(f"## {name or hex_color}")
            st.markdown(ai_output)

    # ========================================================
    # EXPORT SECTION
    # ========================================================
    st.markdown("---")
    st.header("📥 Download Generated Files")

    md_content = generate_markdown(results)
    json_content = generate_json(results)

    md_file, md_name, md_mime = to_downloadable_file(
        md_content, "color_intelligence_report.md", "text/markdown"
    )
    json_file, json_name, json_mime = to_downloadable_file(
        json_content, "color_intelligence_report.json", "application/json"
    )

    st.download_button(
        "Download Markdown Report",
        md_file,
        file_name=md_name,
        mime=md_mime
    )

    st.download_button(
        "Download JSON Report",
        json_file,
        file_name=json_name,
        mime=json_mime
    )

    st.caption("Color Intelligence Platform • Enterprise Demo • v1.0")
