# ============================================================
# Color Intelligence Platform — Enterprise Edition
# Structured JSON Output | Advanced AI | Professional UI
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import colorsys
import json
import re
from io import BytesIO
from transformers import pipeline
from functools import lru_cache
from typing import Dict, List, Any
import time

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Color Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# ENHANCED CSS - PROFESSIONAL RUSTIC THEME
# ============================================================
st.markdown(
    """
    <style>
    /* Global Theme */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Source+Sans+Pro:wght@300;400;600&display=swap');
    
    .main {
        background: linear-gradient(135deg, #f5f1e8 0%, #e8dfd0 100%);
        font-family: 'Source Sans Pro', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Playfair Display', serif;
        color: #3d3027;
        letter-spacing: 0.5px;
    }
    
    /* Header Section */
    .platform-header {
        background: linear-gradient(135deg, #4a3f35 0%, #6b5d52 100%);
        padding: 2.5rem;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(61, 48, 39, 0.3);
        margin-bottom: 2rem;
        border-left: 6px solid #a67c52;
    }
    
    .platform-header h1 {
        color: #f5f1e8;
        margin: 0;
        font-size: 2.8rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .platform-header p {
        color: #d4c4b0;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        font-weight: 300;
    }
    
    /* Color Card */
    .color-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border: 1px solid #d4c4b0;
        margin-bottom: 2rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .color-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
    }
    
    .color-swatch {
        border-radius: 8px;
        border: 3px solid #3d3027;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    
    .color-name {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        color: #3d3027;
        font-weight: 700;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #a67c52;
        padding-bottom: 0.5rem;
    }
    
    .hex-badge {
        display: inline-block;
        background: #4a3f35;
        color: #f5f1e8;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }
    
    /* Feature Grid */
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        margin: 1.5rem 0;
    }
    
    .feature-item {
        background: linear-gradient(135deg, #f9f6f0 0%, #ede7dc 100%);
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #a67c52;
        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    }
    
    .feature-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        color: #6b5d52;
        font-weight: 600;
        letter-spacing: 1px;
        margin-bottom: 0.3rem;
    }
    
    .feature-value {
        font-size: 1.2rem;
        color: #3d3027;
        font-weight: 600;
    }
    
    /* JSON Viewer */
    .json-container {
        background: #2d2d2d;
        border-radius: 8px;
        padding: 1.5rem;
        margin-top: 1.5rem;
        border: 2px solid #4a3f35;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        max-height: 600px;
        overflow-y: auto;
    }
    
    .json-container pre {
        color: #a9dc76;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        margin: 0;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
    
    /* Section Headers */
    .section-header {
        background: linear-gradient(90deg, #6b5d52 0%, #8b7566 100%);
        color: #f5f1e8;
        padding: 1rem 1.5rem;
        border-radius: 8px;
        margin: 2rem 0 1rem 0;
        font-family: 'Playfair Display', serif;
        font-size: 1.4rem;
        font-weight: 700;
        box-shadow: 0 3px 10px rgba(0,0,0,0.15);
        border-left: 5px solid #a67c52;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #a67c52 0%, #c9a66b 100%);
    }
    
    /* Download Buttons */
    .download-section {
        background: #ffffff;
        padding: 2rem;
        border-radius: 12px;
        border: 2px solid #a67c52;
        margin-top: 2rem;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    
    .stDownloadButton button {
        background: linear-gradient(135deg, #4a3f35 0%, #6b5d52 100%);
        color: #f5f1e8;
        border: none;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }
    
    .stDownloadButton button:hover {
        background: linear-gradient(135deg, #6b5d52 0%, #8b7566 100%);
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #4a3f35 0%, #3d3027 100%);
    }
    
    .sidebar .sidebar-content {
        background: #3d3027;
        color: #f5f1e8;
    }
    
    /* Stats Card */
    .stats-card {
        background: linear-gradient(135deg, #a67c52 0%, #c9a66b 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        margin-bottom: 1rem;
    }
    
    .stats-number {
        font-size: 2.5rem;
        font-weight: 700;
        font-family: 'Playfair Display', serif;
    }
    
    .stats-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 0.5rem;
    }
    
    /* Expander Styling */
    .streamlit-expanderHeader {
        background: #f9f6f0;
        border-radius: 8px;
        border: 1px solid #d4c4b0;
        font-weight: 600;
        color: #3d3027;
    }
    
    /* Loading Animation */
    .loading-spinner {
        border: 4px solid #f5f1e8;
        border-top: 4px solid #a67c52;
        border-radius: 50%;
        width: 40px;
        height: 40px;
        animation: spin 1s linear infinite;
        margin: 2rem auto;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f5f1e8;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #a67c52;
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #8b6542;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
    <div class="platform-header">
        <h1>🎨 Color Intelligence Platform</h1>
        <p>Enterprise AI for Color Psychology, Branding & Design Analysis</p>
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# COLOR UTILITIES
# ============================================================
HEX_REGEX = re.compile(r"^#?[0-9A-Fa-f]{6}$")

def normalize_hex(hex_color: str) -> str:
    """Validate and normalize HEX color code"""
    hex_color = hex_color.strip()
    if not HEX_REGEX.match(hex_color):
        raise ValueError(f"Invalid HEX color: {hex_color}")
    if not hex_color.startswith("#"):
        hex_color = "#" + hex_color
    return hex_color.upper()

def hex_to_rgb(hex_color):
    """Convert HEX to RGB"""
    hex_color = hex_color.lstrip("#")
    return np.array([
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16)
    ])

def rgb_to_lab(rgb):
    """Convert RGB to LAB color space for perceptual accuracy"""
    # Normalize RGB to 0-1
    rgb = rgb / 255.0
    
    # Convert to XYZ
    def rgb_to_xyz_channel(c):
        if c > 0.04045:
            return ((c + 0.055) / 1.055) ** 2.4
        else:
            return c / 12.92
    
    rgb = np.array([rgb_to_xyz_channel(c) for c in rgb])
    
    # XYZ matrix transformation (sRGB to XYZ)
    xyz = np.dot(rgb, np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ]).T)
    
    # Normalize by D65 white point
    xyz = xyz / np.array([0.95047, 1.00000, 1.08883])
    
    # Convert to LAB
    def xyz_to_lab_channel(t):
        if t > 0.008856:
            return t ** (1/3)
        else:
            return (7.787 * t) + (16/116)
    
    xyz = np.array([xyz_to_lab_channel(c) for c in xyz])
    
    L = (116 * xyz[1]) - 16
    a = 500 * (xyz[0] - xyz[1])
    b = 200 * (xyz[1] - xyz[2])
    
    return np.array([L, a, b])

def delta_e(lab1, lab2):
    """Calculate Delta E (CIE76) - perceptual color difference"""
    return np.sqrt(np.sum((lab1 - lab2) ** 2))

def extract_features(hex_color):
    """Extract comprehensive color features"""
    r, g, b = hex_to_rgb(hex_color) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    
    # Temperature classification
    hue_deg = h * 360
    if hue_deg < 30 or hue_deg > 330:
        temp = "Warm (Red-based)"
    elif 30 <= hue_deg < 90:
        temp = "Warm (Yellow-based)"
    elif 90 <= hue_deg < 150:
        temp = "Cool (Green-based)"
    elif 150 <= hue_deg < 270:
        temp = "Cool (Blue-based)"
    else:
        temp = "Warm (Purple-based)"
    
    return {
        "hex": hex_color,
        "hue_deg": round(hue_deg, 2),
        "lightness_pct": round(l * 100, 2),
        "saturation_pct": round(s * 100, 2),
        "temperature": temp,
        "rgb": f"rgb({int(r*255)}, {int(g*255)}, {int(b*255)})"
    }

def calculate_contrast_ratio(hex1, hex2="#FFFFFF"):
    """Calculate WCAG contrast ratio"""
    def relative_luminance(rgb):
        rgb = rgb / 255.0
        rgb = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
        return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
    
    lum1 = relative_luminance(hex_to_rgb(hex1))
    lum2 = relative_luminance(hex_to_rgb(hex2))
    
    lighter = max(lum1, lum2)
    darker = min(lum1, lum2)
    
    return round((lighter + 0.05) / (darker + 0.05), 2)

def accessibility_analysis(hex_color):
    """Generate comprehensive accessibility analysis"""
    features = extract_features(hex_color)
    l = features["lightness_pct"]
    
    # Contrast ratios
    contrast_white = calculate_contrast_ratio(hex_color, "#FFFFFF")
    contrast_black = calculate_contrast_ratio(hex_color, "#000000")
    
    analysis = []
    
    # Lightness assessment
    if l < 25:
        analysis.append(f"Very dark tone ({l}% lightness). Excellent for backgrounds with light text.")
    elif l < 45:
        analysis.append(f"Dark tone ({l}% lightness). Suitable for headings and emphasis elements.")
    elif l < 70:
        analysis.append(f"Medium tone ({l}% lightness). Versatile for UI elements and accents.")
    else:
        analysis.append(f"Light tone ({l}% lightness). Best used on dark backgrounds for proper contrast.")
    
    # WCAG compliance
    if contrast_white >= 7.0:
        analysis.append(f"AAA compliant with white text (contrast: {contrast_white}:1). Excellent for body text.")
    elif contrast_white >= 4.5:
        analysis.append(f"AA compliant with white text (contrast: {contrast_white}:1). Suitable for body text.")
    elif contrast_white >= 3.0:
        analysis.append(f"AA compliant for large text only (contrast: {contrast_white}:1).")
    else:
        analysis.append(f"Low contrast with white text ({contrast_white}:1). Use black text instead.")
    
    if contrast_black >= 7.0:
        analysis.append(f"AAA compliant with black text (contrast: {contrast_black}:1).")
    elif contrast_black >= 4.5:
        analysis.append(f"AA compliant with black text (contrast: {contrast_black}:1).")
    
    # Recommendations
    if features["saturation_pct"] > 70:
        analysis.append("High saturation may cause eye strain in large areas. Use sparingly for accents.")
    
    return analysis

def compute_similar_colors(target_hex, all_hexes, top_n=5):
    """Find similar colors using perceptual LAB color space"""
    target_lab = rgb_to_lab(hex_to_rgb(target_hex))
    
    distances = []
    for h in all_hexes:
        if h == target_hex:
            continue
        lab = rgb_to_lab(hex_to_rgb(h))
        dist = delta_e(target_lab, lab)
        distances.append((h, dist))
    
    distances.sort(key=lambda x: x[1])
    return [h for h, _ in distances[:top_n]]

# ============================================================
# ADVANCED LLM PROMPT - STRUCTURED JSON OUTPUT
# ============================================================
MASTER_PROMPT = """You are an elite color theorist, brand strategist, and design historian specializing in color psychology and accessibility. Your expertise spans historical color usage, cultural symbolism, emotional psychology, and modern design applications.

CRITICAL INSTRUCTIONS:
- Output ONLY valid JSON. No preamble, no explanations, no markdown formatting.
- Use sophisticated, professional language befitting a premium design platform
- Be specific and actionable in all recommendations
- Draw from real historical, cultural, and psychological research
- Avoid clichés, emojis, and marketing fluff
- Ensure all arrays contain exactly 4 items for consistency
- Make each description unique and insightful

INPUT DATA:
HEX: {hex_code}
Hue: {hue}°
Lightness: {lightness}%
Saturation: {saturation}%
Temperature: {temperature}

OUTPUT STRUCTURE (STRICT JSON):
{{
  "short_description": "A single evocative sentence (15-25 words) capturing the essence and visual character of this color.",
  "description": [
    "First paragraph: Detailed sensory description - how the color appears visually, its depth, tone, and visual texture.",
    "Second paragraph: Emotional and atmospheric qualities - the mood and feelings it evokes in viewers.",
    "Third paragraph: Contextual positioning - where this color sits in the spectrum and what makes it distinctive."
  ],
  "psychology": {{
    "overview": "Comprehensive overview (40-60 words) of this color's psychological impact on human perception and behavior.",
    "associations": [
      {{"trait": "Primary emotional association", "meaning": "Detailed explanation of why this color evokes this emotion"}},
      {{"trait": "Cognitive association", "meaning": "How this color affects thinking patterns and mental processes"}},
      {{"trait": "Behavioral association", "meaning": "Actions or behaviors this color tends to encourage"}},
      {{"trait": "Cultural association", "meaning": "Broader cultural or social meanings attached to this color"}}
    ],
    "impact": "Summary (30-45 words) of overall psychological effect and when to leverage these properties."
  }},
  "meaning": {{
    "overview": "Cultural and symbolic significance (40-60 words) - what this color represents across contexts.",
    "themes": [
      {{"concept": "Primary symbolic theme", "description": "Deep dive into this symbolic meaning"}},
      {{"concept": "Secondary symbolic theme", "description": "Additional layer of symbolic interpretation"}},
      {{"concept": "Historical symbolism", "description": "Traditional or historical symbolic associations"}},
      {{"concept": "Modern interpretation", "description": "Contemporary symbolic meanings and connotations"}}
    ],
    "note": "Nuanced observation (25-40 words) about how this color's meaning varies by context or culture."
  }},
  "why_use_this_color": {{
    "purpose": "Strategic overview (35-50 words) of when and why designers should choose this color.",
    "benefits": [
      {{"use": "Brand identity application", "reason": "Specific benefit for brand/logo usage with concrete examples"}},
      {{"use": "User interface design", "reason": "Specific benefit for UI/UX applications"}},
      {{"use": "Marketing & communication", "reason": "Specific benefit for marketing materials and messaging"}},
      {{"use": "Environmental design", "reason": "Specific benefit for physical spaces and product design"}}
    ],
    "summary": "Closing insight (30-45 words) on this color's unique value proposition in design."
  }},
  "applications": {{
    "guidance": "Practical guidance (40-60 words) on best practices for implementing this color effectively.",
    "examples": [
      {{"domain": "Digital/Web Design", "usage": "Specific application examples with actionable tips"}},
      {{"domain": "Print & Branding", "usage": "Specific application examples for print materials"}},
      {{"domain": "Interior/Product Design", "usage": "Specific application examples for physical design"}},
      {{"domain": "Fashion & Textiles", "usage": "Specific application examples for apparel and fabric"}}
    ],
    "note": "Advanced tip (25-40 words) for professional designers on maximizing this color's impact."
  }},
  "history": [
    "Historical paragraph 1: Ancient and classical usage - specific civilizations, periods, and cultural contexts.",
    "Historical paragraph 2: Evolution through Renaissance, Industrial age, and early modern periods.",
    "Historical paragraph 3: Modern history - 20th/21st century usage, design movements, and contemporary significance."
  ],
  "accessibility": [
    "Contrast and readability analysis with specific WCAG guidelines",
    "Recommendations for text color pairings (light/dark backgrounds)",
    "Considerations for color-blind users and perception variations",
    "Best practices for inclusive design with this color"
  ]
}}

Generate the JSON now:"""

# ============================================================
# LOAD LLM
# ============================================================
@st.cache_resource
def load_llm():
    """Load local LLM with optimized settings"""
    return pipeline(
        "text2text-generation",
        model="google/flan-t5-large",
        max_length=2048,  # Increased for JSON output
        temperature=0.7,  # Balanced creativity
        top_p=0.9,
        do_sample=True
    )

try:
    llm = load_llm()
except Exception as e:
    st.error(f"Error loading AI model: {e}")
    st.stop()

# ============================================================
# LLM GENERATION WITH JSON PARSING
# ============================================================
@lru_cache(maxsize=512)
def generate_color_profile(hex_code: str, hue: float, lightness: float, saturation: float, temperature: str) -> Dict[str, Any]:
    """Generate structured JSON color profile using AI"""
    
    prompt = MASTER_PROMPT.format(
        hex_code=hex_code,
        hue=hue,
        lightness=lightness,
        saturation=saturation,
        temperature=temperature
    )
    
    try:
        # Generate response
        output = llm(prompt)[0]["generated_text"]
        
        # Clean and parse JSON
        output = output.strip()
        
        # Remove markdown code blocks if present
        if output.startswith("```"):
            output = re.sub(r'^```(?:json)?\s*', '', output)
            output = re.sub(r'\s*```$', '', output)
        
        # Parse JSON
        parsed_json = json.loads(output)
        
        return parsed_json
        
    except json.JSONDecodeError as e:
        # Fallback structure if JSON parsing fails
        st.warning(f"AI output parsing issue for {hex_code}. Using fallback structure.")
        return create_fallback_profile(hex_code, hue, lightness, saturation, temperature)
    except Exception as e:
        st.error(f"AI generation error for {hex_code}: {e}")
        return create_fallback_profile(hex_code, hue, lightness, saturation, temperature)

def create_fallback_profile(hex_code: str, hue: float, lightness: float, saturation: float, temperature: str) -> Dict[str, Any]:
    """Create deterministic fallback profile when AI fails"""
    
    # Determine color family
    if hue < 30 or hue > 330:
        family = "red"
        family_name = "Red"
    elif hue < 90:
        family = "yellow"
        family_name = "Yellow"
    elif hue < 150:
        family = "green"
        family_name = "Green"
    elif hue < 210:
        family = "cyan"
        family_name = "Cyan"
    elif hue < 270:
        family = "blue"
        family_name = "Blue"
    else:
        family = "purple"
        family_name = "Purple"
    
    tone = "light" if lightness > 60 else ("dark" if lightness < 40 else "medium")
    
    return {
        "short_description": f"A {tone} {family_name.lower()} tone with {saturation:.0f}% saturation, offering {temperature.lower()} visual characteristics.",
        "description": [
            f"This color presents as a {tone} shade within the {family_name.lower()} spectrum, featuring {saturation:.0f}% saturation and {lightness:.0f}% lightness.",
            f"The {temperature.lower()} temperature creates a {'energetic' if 'Warm' in temperature else 'calming'} visual presence.",
            f"Positioned at {hue:.0f}° on the color wheel, this hue occupies a distinctive place in the {family_name.lower()} range."
        ],
        "psychology": {
            "overview": f"This {family_name.lower()} tone evokes associations common to its color family while its specific characteristics influence viewer perception.",
            "associations": [
                {"trait": "Emotional response", "meaning": f"Tends to evoke {family_name.lower()}-associated emotions"},
                {"trait": "Cognitive impact", "meaning": "Influences attention and focus based on saturation level"},
                {"trait": "Behavioral effect", "meaning": f"{'Stimulating' if saturation > 60 else 'Subdued'} behavioral response"},
                {"trait": "Cultural meaning", "meaning": f"Carries traditional {family_name.lower()} symbolism"}
            ],
            "impact": f"Overall psychological effect is {'vibrant and attention-grabbing' if saturation > 60 else 'subtle and harmonious'}."
        },
        "meaning": {
            "overview": f"Symbolically represents themes associated with {family_name.lower()} hues across various cultural contexts.",
            "themes": [
                {"concept": "Primary theme", "description": f"Traditional {family_name.lower()} symbolism"},
                {"concept": "Secondary theme", "description": f"Nuanced by its {tone} character"},
                {"concept": "Historical context", "description": f"{family_name} has rich historical usage"},
                {"concept": "Modern interpretation", "description": "Contemporary design applications"}
            ],
            "note": "Symbolic meaning varies significantly based on cultural context and application."
        },
        "why_use_this_color": {
            "purpose": f"Effective for applications requiring {family_name.lower()} color psychology and {tone} visual weight.",
            "benefits": [
                {"use": "Branding", "reason": f"Leverages {family_name.lower()} associations"},
                {"use": "UI Design", "reason": f"{'High visibility' if lightness < 50 and saturation > 60 else 'Harmonious integration'}"},
                {"use": "Marketing", "reason": f"Communicates through {family_name.lower()} symbolism"},
                {"use": "Product Design", "reason": f"{tone.capitalize()} tone offers versatile application"}
            ],
            "summary": f"Provides reliable {family_name.lower()}-family characteristics with {tone} tonal properties."
        },
        "applications": {
            "guidance": f"Best applied where {family_name.lower()} psychology and {tone} contrast are desired.",
            "examples": [
                {"domain": "Web Design", "usage": f"Suitable for {tone}-background applications"},
                {"domain": "Print", "usage": f"Works well in {tone} design contexts"},
                {"domain": "Interior", "usage": f"{family_name} accents with {tone} character"},
                {"domain": "Fashion", "usage": f"Versatile {family_name.lower()} option"}
            ],
            "note": f"Consider pairing with complementary tones for optimal visual impact."
        },
        "history": [
            f"{family_name} pigments have been used since ancient times across various civilizations.",
            f"The specific tone and saturation became more achievable with modern color technology.",
            f"Contemporary usage reflects both traditional symbolism and modern aesthetic trends."
        ],
        "accessibility": [
            f"Lightness of {lightness:.0f}% affects text readability and contrast requirements.",
            f"Pair with {'light' if lightness < 50 else 'dark'} text for optimal readability.",
            f"Consider color-blind simulation for {family_name.lower()} hues.",
            "Follow WCAG guidelines for contrast ratios in text applications."
        ]
    }

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.markdown("### ⚙️ Platform Settings")
st.sidebar.markdown("**AI Model:** FLAN-T5-Large (Local)")
st.sidebar.markdown("**Color Space:** LAB (Perceptual)")
st.sidebar.markdown("**Output Format:** Structured JSON")
st.sidebar.markdown("**Theme:** Rustic Professional")
st.sidebar.markdown("---")

show_json_preview = st.sidebar.checkbox("Show JSON Preview", value=True)
show_features = st.sidebar.checkbox("Show Color Features", value=True)
show_accessibility = st.sidebar.checkbox("Show Accessibility", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Color Intelligence Platform")

# ============================================================
# SAMPLE CSV
# ============================================================
SAMPLE_CSV = """hex,name
#AB274F,Cherry Rose
#8B2E3C,Rust Wine
#C9A66B,Antique Gold
#5E3A2E,Chestnut Bark
#2F3E46,Deep Slate
#A67C52,Old Leather
#4A3F35,Espresso Brown
#6B5D52,Warm Taupe"""

# ============================================================
# FILE UPLOAD SECTION
# ============================================================
st.markdown('<div class="section-header">📁 Upload Color Dataset</div>', unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload CSV file with HEX colors",
        type=["csv"],
        help="CSV must contain a 'hex' column. Optional 'name' column for color labels."
    )

with col2:
    st.download_button(
        label="📥 Download Sample CSV",
        data=SAMPLE_CSV,
        file_name="sample_colors.csv",
        mime="text/csv",
        help="Template file to get started"
    )

# ============================================================
# MAIN PROCESSING LOGIC
# ============================================================
if uploaded_file:
    try:
        # Load CSV
        df = pd.read_csv(uploaded_file)
        
        # Validate columns
        if "hex" not in df.columns:
            st.error("❌ CSV must contain a 'hex' column")
            st.stop()
        
        # Normalize HEX codes
        try:
            df["hex"] = df["hex"].apply(normalize_hex)
        except ValueError as e:
            st.error(f"❌ Invalid HEX code found: {e}")
            st.stop()
        
        # Handle name column
        if "name" not in df.columns:
            df["name"] = df["hex"]
        else:
            df["name"] = df["name"].fillna(df["hex"])
        
        # Success message
        st.success(f"✅ Successfully loaded {len(df)} colors")
        
        # Display stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(
                f"""
                <div class="stats-card">
                    <div class="stats-number">{len(df)}</div>
                    <div class="stats-label">Total Colors</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"""
                <div class="stats-card">
                    <div class="stats-number">{len(df['name'].unique())}</div>
                    <div class="stats-label">Unique Names</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        with col3:
            st.markdown(
                f"""
                <div class="stats-card">
                    <div class="stats-number">JSON</div>
                    <div class="stats-label">Output Format</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # ============================================================
        # PROCESS AND DISPLAY COLORS ONE BY ONE
        # ============================================================
        st.markdown('<div class="section-header">🎨 Processing & Analyzing Colors</div>', unsafe_allow_html=True)
        
        all_hexes = df["hex"].tolist()
        results = []
        
        # Create a container for overall progress
        progress_container = st.container()
        with progress_container:
            overall_progress = st.progress(0)
            overall_status = st.empty()
        
        st.markdown("---")
        
        # Process each color and display results immediately
        for idx, row in df.iterrows():
            hex_color = row["hex"]
            name = row["name"]
            
            # Update overall progress
            progress = (idx + 1) / len(df)
            overall_progress.progress(progress)
            overall_status.markdown(f"**Processing:** {name} ({idx + 1}/{len(df)})")
            
            # Create a card for this color
            st.markdown(
                f"""
                <div class="section-header" style="background: linear-gradient(90deg, {hex_color} 0%, {hex_color}dd 100%); 
                     color: {'#ffffff' if extract_features(hex_color)['lightness_pct'] < 50 else '#000000'};">
                    🎨 Processing: {name}
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Create processing status for this color
            color_status = st.empty()
            
            # Extract features
            color_status.info("⚙️ Extracting color features...")
            features = extract_features(hex_color)
            time.sleep(0.3)  # Brief pause for visual feedback
            
            # Accessibility analysis
            color_status.info("♿ Analyzing accessibility...")
            accessibility = accessibility_analysis(hex_color)
            time.sleep(0.3)
            
            # Find similar colors
            color_status.info("🔍 Finding similar colors...")
            similar = compute_similar_colors(hex_color, all_hexes, top_n=5)
            time.sleep(0.3)
            
            # Generate AI profile
            color_status.info("🤖 Generating AI analysis (this may take a moment)...")
            ai_profile = generate_color_profile(
                hex_color,
                features["hue_deg"],
                features["lightness_pct"],
                features["saturation_pct"],
                features["temperature"]
            )
            
            # Clear status
            color_status.success(f"✅ {name} processed successfully!")
            time.sleep(0.5)
            color_status.empty()
            
            # Combine everything
            color_data = {
                "name": name,
                "hex": hex_color,
                "features": features,
                "accessibility": accessibility,
                "similar_colors": similar,
                "ai_profile": ai_profile
            }
            
            results.append(color_data)
            
            # ============================================================
            # DISPLAY RESULTS FOR THIS COLOR IMMEDIATELY
            # ============================================================
            
            with st.container():
                col1, col2 = st.columns([1, 2])
                
                # LEFT COLUMN: Visual & Features
                with col1:
                    # Color swatch
                    st.markdown(
                        f"""
                        <div class="color-swatch" style="background: {hex_color}; height: 200px; width: 100%;"></div>
                        <div class="hex-badge">{hex_color}</div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    # Features grid
                    if show_features:
                        st.markdown("**Color Properties**")
                        st.markdown(
                            f"""
                            <div class="feature-grid">
                                <div class="feature-item">
                                    <div class="feature-label">Hue</div>
                                    <div class="feature-value">{features['hue_deg']}°</div>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-label">Lightness</div>
                                    <div class="feature-value">{features['lightness_pct']}%</div>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-label">Saturation</div>
                                    <div class="feature-value">{features['saturation_pct']}%</div>
                                </div>
                                <div class="feature-item">
                                    <div class="feature-label">Temperature</div>
                                    <div class="feature-value" style="font-size: 0.9rem;">{features['temperature']}</div>
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                    
                    # Accessibility
                    if show_accessibility:
                        st.markdown("**Accessibility Notes**")
                        for note in accessibility:
                            st.caption(f"• {note}")
                    
                    # Similar colors
                    if similar:
                        st.markdown("**Similar Colors**")
                        similar_html = ""
                        for sim_hex in similar[:3]:
                            similar_html += f'<div style="display: inline-block; width: 50px; height: 50px; background: {sim_hex}; margin: 5px; border-radius: 6px; border: 2px solid #3d3027;" title="{sim_hex}"></div>'
                        st.markdown(similar_html, unsafe_allow_html=True)
                
                # RIGHT COLUMN: AI Profile
                with col2:
                    st.markdown(f'<div class="color-name">{name}</div>', unsafe_allow_html=True)
                    
                    # Short description
                    if "short_description" in ai_profile:
                        st.markdown(f"*{ai_profile['short_description']}*")
                        st.markdown("---")
                    
                    # Description
                    if "description" in ai_profile and isinstance(ai_profile["description"], list):
                        st.markdown("**Description**")
                        for para in ai_profile["description"]:
                            st.write(para)
                    
                    # Psychology Section
                    if "psychology" in ai_profile:
                        with st.expander("🧠 Psychology", expanded=False):
                            psych = ai_profile["psychology"]
                            if "overview" in psych:
                                st.write(psych["overview"])
                            if "associations" in psych and isinstance(psych["associations"], list):
                                st.markdown("**Key Associations:**")
                                for assoc in psych["associations"]:
                                    if isinstance(assoc, dict) and "trait" in assoc and "meaning" in assoc:
                                        st.markdown(f"- **{assoc['trait']}:** {assoc['meaning']}")
                            if "impact" in psych:
                                st.info(psych["impact"])
                    
                    # Meaning Section
                    if "meaning" in ai_profile:
                        with st.expander("💡 Meaning & Symbolism", expanded=False):
                            meaning = ai_profile["meaning"]
                            if "overview" in meaning:
                                st.write(meaning["overview"])
                            if "themes" in meaning and isinstance(meaning["themes"], list):
                                st.markdown("**Symbolic Themes:**")
                                for theme in meaning["themes"]:
                                    if isinstance(theme, dict) and "concept" in theme and "description" in theme:
                                        st.markdown(f"- **{theme['concept']}:** {theme['description']}")
                            if "note" in meaning:
                                st.caption(meaning["note"])
                    
                    # Why Use Section
                    if "why_use_this_color" in ai_profile:
                        with st.expander("✨ Why Use This Color", expanded=False):
                            why_use = ai_profile["why_use_this_color"]
                            if "purpose" in why_use:
                                st.write(why_use["purpose"])
                            if "benefits" in why_use and isinstance(why_use["benefits"], list):
                                st.markdown("**Key Benefits:**")
                                for benefit in why_use["benefits"]:
                                    if isinstance(benefit, dict) and "use" in benefit and "reason" in benefit:
                                        st.markdown(f"- **{benefit['use']}:** {benefit['reason']}")
                            if "summary" in why_use:
                                st.success(why_use["summary"])
                    
                    # Applications Section
                    if "applications" in ai_profile:
                        with st.expander("🎯 Applications", expanded=False):
                            apps = ai_profile["applications"]
                            if "guidance" in apps:
                                st.write(apps["guidance"])
                            if "examples" in apps and isinstance(apps["examples"], list):
                                st.markdown("**Domain Examples:**")
                                for example in apps["examples"]:
                                    if isinstance(example, dict) and "domain" in example and "usage" in example:
                                        st.markdown(f"- **{example['domain']}:** {example['usage']}")
                            if "note" in apps:
                                st.info(apps["note"])
                    
                    # History Section
                    if "history" in ai_profile and isinstance(ai_profile["history"], list):
                        with st.expander("📜 History", expanded=False):
                            for hist_para in ai_profile["history"]:
                                st.write(hist_para)
                    
                    # Show full JSON
                    if show_json_preview:
                        with st.expander("📋 Complete AI Analysis (JSON)", expanded=False):
                            st.markdown(
                                f"""
                                <div class="json-container">
                                    <pre>{json.dumps(ai_profile, indent=2)}</pre>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
            
            # Separator between colors
            st.markdown("---")
        
        # Clear overall progress
        overall_progress.empty()
        overall_status.empty()
        
        # Final success message
        st.success(f"🎉 All {len(results)} colors processed and displayed successfully!")
        
        # ============================================================
        # EXPORT SECTION
        # ============================================================
        st.markdown('<div class="section-header">💾 Export Options</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="download-section">', unsafe_allow_html=True)
        
        # Prepare export data
        export_data = {
            "metadata": {
                "total_colors": len(results),
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "platform": "Color Intelligence Platform",
                "version": "2.0"
            },
            "colors": []
        }
        
        for color_data in results:
            export_data["colors"].append({
                "name": color_data["name"],
                "hex": color_data["hex"],
                "rgb": color_data["features"]["rgb"],
                "properties": {
                    "hue": color_data["features"]["hue_deg"],
                    "lightness": color_data["features"]["lightness_pct"],
                    "saturation": color_data["features"]["saturation_pct"],
                    "temperature": color_data["features"]["temperature"]
                },
                "analysis": color_data["ai_profile"],
                "accessibility": color_data["accessibility"],
                "similar_colors": color_data["similar_colors"]
            })
        
        # JSON export
        json_output = json.dumps(export_data, indent=2)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📥 Download Complete JSON",
                data=json_output,
                file_name=f"color_intelligence_export_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Complete dataset with all AI analysis in structured JSON format"
            )
        
        with col2:
            # Individual color JSONs (ZIP would be ideal, but keeping it simple)
            individual_json = json.dumps([c["ai_profile"] for c in results], indent=2)
            st.download_button(
                label="📥 Download AI Profiles Only",
                data=individual_json,
                file_name=f"ai_profiles_{time.strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                help="Only the AI-generated color profiles"
            )
        
        with col3:
            # Markdown report
            md_report = f"# Color Intelligence Report\n\n"
            md_report += f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            md_report += f"**Total Colors:** {len(results)}\n\n"
            md_report += "---\n\n"
            
            for color_data in results:
                md_report += f"## {color_data['name']}\n\n"
                md_report += f"**HEX:** `{color_data['hex']}`\n\n"
                md_report += f"**RGB:** `{color_data['features']['rgb']}`\n\n"
                
                profile = color_data['ai_profile']
                if 'short_description' in profile:
                    md_report += f"*{profile['short_description']}*\n\n"
                
                if 'description' in profile:
                    for para in profile['description']:
                        md_report += f"{para}\n\n"
                
                md_report += "---\n\n"
            
            st.download_button(
                label="📥 Download Markdown Report",
                data=md_report,
                file_name=f"color_report_{time.strftime('%Y%m%d_%H%M%S')}.md",
                mime="text/markdown",
                help="Human-readable markdown report"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"❌ Error processing file: {e}")
        st.exception(e)

else:
    # ============================================================
    # WELCOME SCREEN
    # ============================================================
    st.markdown(
        """
        <div class="color-card" style="text-align: center; padding: 3rem;">
            <h2 style="color: #3d3027; margin-bottom: 1rem;">Welcome to Color Intelligence Platform</h2>
            <p style="font-size: 1.1rem; color: #6b5d52; margin-bottom: 2rem;">
                Upload your CSV file to begin professional color analysis powered by AI
            </p>
            <div style="background: linear-gradient(135deg, #a67c52 0%, #c9a66b 100%); 
                        padding: 2rem; border-radius: 10px; margin: 2rem auto; max-width: 600px;">
                <h3 style="color: #ffffff; margin-bottom: 1rem;">Platform Features</h3>
                <ul style="color: #f5f1e8; text-align: left; line-height: 2;">
                    <li>🎨 Advanced color psychology analysis</li>
                    <li>🤖 AI-powered structured JSON generation</li>
                    <li>♿ Comprehensive accessibility evaluation</li>
                    <li>🔍 Perceptual color similarity matching (LAB space)</li>
                    <li>📊 WCAG contrast ratio calculations</li>
                    <li>💾 Professional export formats (JSON, Markdown)</li>
                    <li>🎯 Batch processing for enterprise datasets</li>
                </ul>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Instructions
    st.markdown("### 📋 Getting Started")
    st.markdown(
        """
        1. **Download** the sample CSV template using the button above
        2. **Add your colors** in HEX format (e.g., `#AB274F`)
        3. **Optionally add names** for each color in the 'name' column
        4. **Upload** your CSV file
        5. **Review** the AI-generated analysis
        6. **Export** your results in JSON or Markdown format
        """
    )
    
    st.markdown("### 🎯 Output Format")
    st.code(
        """{
  "short_description": "Evocative single sentence",
  "description": ["Visual description", "Emotional qualities", "Contextual positioning"],
  "psychology": {
    "overview": "Psychological impact overview",
    "associations": [
      {"trait": "...", "meaning": "..."}
    ],
    "impact": "Summary of psychological effect"
  },
  "meaning": { ... },
  "why_use_this_color": { ... },
  "applications": { ... },
  "history": [ ... ],
  "accessibility": [ ... ]
}""",
        language="json"
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Color Intelligence Platform v2.0 • Enterprise AI for Color Analysis • Powered by FLAN-T5")