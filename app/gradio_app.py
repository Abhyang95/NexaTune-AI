import html
import requests
import gradio as gr


# ============================================================
# CONFIG
# ============================================================

API_URL = "http://127.0.0.1:8000"


# ============================================================
# REAL BENCHMARK DATA
# ============================================================

BASE_ROUGE = 0.2120
NEXA_ROUGE = 0.4082

BASE_BERT = 0.8574
NEXA_BERT = 0.9160

BASE_LATENCY = 9.22
NEXA_LATENCY = 15.45

BASE_TOKENS = 100.1
NEXA_TOKENS = 99.9

ROUGE_DELTA = 0.1962
BERT_DELTA = 0.0585

ROUGE_CI = "[+0.1751, +0.2185]"
BERT_CI = "[+0.0535, +0.0638]"

AI_MARGIN = 0.12
AI_CI = "[−0.21, +0.49]"


# ============================================================
# PREMIUM UI CSS
# ============================================================

CSS = """

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');


/* ==========================================================
   GLOBAL
   ========================================================== */

html {
    scroll-behavior: smooth;
}

body {
    background: #080808 !important;
}

.gradio-container {
    max-width: 1500px !important;
    margin: auto !important;
    background:
        radial-gradient(
            circle at 85% 0%,
            rgba(255,255,255,0.035),
            transparent 28%
        ),
        radial-gradient(
            circle at 0% 35%,
            rgba(255,255,255,0.018),
            transparent 32%
        ),
        #080808 !important;

    color: #f4f4f4 !important;
    font-family: "Inter", sans-serif !important;
}

footer {
    display: none !important;
}

* {
    box-sizing: border-box;
}

button,
textarea,
input {
    font-family: "Inter", sans-serif !important;
}


/* ==========================================================
   NAVBAR
   ========================================================== */

.top-nav {
    position: sticky;
    top: 0;
    z-index: 1000;

    display: flex;
    align-items: center;
    justify-content: space-between;

    padding: 16px 10px;

    background: rgba(8,8,8,0.88);
    backdrop-filter: blur(18px);
    -webkit-backdrop-filter: blur(18px);

    border-bottom: 1px solid #242424;
}

.brand {
    display: flex;
    align-items: center;
    gap: 11px;

    min-width: 170px;
}

.brand-mark {
    width: 31px;
    height: 31px;

    display: flex;
    align-items: center;
    justify-content: center;

    border: 1px solid #343434;
    border-radius: 8px;

    background:
        linear-gradient(
            145deg,
            #1b1b1b,
            #0c0c0c
        );

    color: #f4f4f4;

    font-family: "Space Grotesk", sans-serif;
    font-size: 11px;
    font-weight: 700;

    box-shadow:
        inset 0 1px rgba(255,255,255,0.05),
        0 5px 20px rgba(0,0,0,.2);
}

.brand-name {
    font-family: "Space Grotesk", sans-serif;
    font-weight: 600;
    font-size: 15px;
    letter-spacing: -0.025em;
}

.nav-links {
    display: flex;
    align-items: center;
    justify-content: center;

    gap: 5px;
}

.nav-link {
    color: #858585 !important;
    text-decoration: none !important;

    padding: 7px 10px;

    border-radius: 7px;

    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: .09em;

    transition:
        color .2s ease,
        background .2s ease;
}

.nav-link:hover {
    color: #eeeeee !important;
    background: #151515;
}

.nav-meta {
    min-width: 170px;

    display: flex;
    align-items: center;
    justify-content: flex-end;

    gap: 8px;

    color: #707070;

    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .12em;
}

.status-dot {
    width: 6px;
    height: 6px;

    border-radius: 50%;

    background: #8ce0d2;

    box-shadow:
        0 0 12px rgba(140,224,210,.55);
}


/* ==========================================================
   HERO
   ========================================================== */

.hero {
    padding: 74px 4px 65px;
}

.hero-grid {
    display: grid;
    grid-template-columns: 1.25fr .75fr;

    gap: 65px;

    align-items: end;
}

.kicker {
    margin-bottom: 17px;

    color: #8ce0d2;

    font-size: 10px;
    font-weight: 600;

    text-transform: uppercase;
    letter-spacing: .18em;
}

.hero-title {
    margin: 0;

    max-width: 850px;

    font-family: "Space Grotesk", sans-serif;

    font-size: clamp(48px, 6.3vw, 82px);

    line-height: .95;

    letter-spacing: -.06em;

    font-weight: 600;
}

.hero-copy {
    max-width: 720px;

    margin-top: 26px;

    color: #969696;

    font-size: 14px;
    line-height: 1.85;
}

.hero-readout {
    padding-left: 27px;

    border-left: 1px solid #333;
}

.readout-label {
    margin-bottom: 14px;

    color: #696969;

    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .16em;
}

.readout-main {
    margin-bottom: 7px;

    color: #eeeeee;

    font-family: "Space Grotesk", sans-serif;
    font-size: 31px;

    letter-spacing: -.04em;
}

.readout-sub {
    color: #8c8c8c;

    font-size: 11px;
    line-height: 1.8;
}


/* ==========================================================
   SECTION
   ========================================================== */

.section {
    margin-top: 70px;

    scroll-margin-top: 95px;
}

.section-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;

    margin-bottom: 22px;

    padding-bottom: 13px;

    border-bottom: 1px solid #272727;
}

.section-title {
    color: #eeeeee;

    font-family: "Space Grotesk", sans-serif;
    font-size: 20px;
    font-weight: 600;

    letter-spacing: -.025em;
}

.section-label {
    color: #666666;

    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .15em;
}


/* ==========================================================
   WORKSPACE
   ========================================================== */

.workspace {
    scroll-margin-top: 95px;

    padding: 22px !important;

    border: 1px solid #292929 !important;
    border-radius: 16px !important;

    background:
        linear-gradient(
            145deg,
            rgba(20,20,20,.86),
            rgba(10,10,10,.94)
        ) !important;

    box-shadow:
        0 20px 70px rgba(0,0,0,.2),
        inset 0 1px rgba(255,255,255,.025);
}

.workspace-grid {
    display: grid;

    grid-template-columns: 1fr 1fr;

    gap: 19px;
}

.workspace-panel {
    min-width: 0;

    padding: 20px;

    border: 1px solid #292929;

    border-radius: 13px;

    background: #101010;

    transition:
        border-color .25s ease,
        transform .25s ease,
        background .25s ease;
}

.workspace-panel:hover {
    border-color: #3a3a3a;

    background: #121212;

    transform: translateY(-2px);
}

.panel-label {
    margin-bottom: 12px;

    color: #676767;

    font-size: 9px;

    text-transform: uppercase;
    letter-spacing: .15em;
}


/* ==========================================================
   GRADIO INPUT OVERRIDES
   ========================================================== */

textarea {
    background: #0c0c0c !important;

    color: #e9e9e9 !important;

    border: 1px solid #292929 !important;

    border-radius: 10px !important;

    line-height: 1.65 !important;

    font-size: 13px !important;

    transition:
        border-color .2s ease,
        box-shadow .2s ease !important;
}

textarea:focus {
    border-color: #4b4b4b !important;

    box-shadow:
        0 0 0 1px rgba(255,255,255,.035) !important;
}

textarea::placeholder {
    color: #5d5d5d !important;
}


/* ==========================================================
   QUICK PROMPTS
   ========================================================== */

.quick-row {
    display: flex !important;

    flex-wrap: wrap !important;

    gap: 7px !important;

    margin-top: 12px !important;
}

.quick-btn {
    min-width: auto !important;

    border: 1px solid #303030 !important;

    border-radius: 999px !important;

    background: #111111 !important;

    color: #a4a4a4 !important;

    font-size: 10px !important;

    transition:
        transform .2s ease,
        border-color .2s ease,
        color .2s ease,
        background .2s ease !important;
}

.quick-btn:hover {
    transform: translateY(-2px);

    border-color: #505050 !important;

    background: #181818 !important;

    color: #f0f0f0 !important;
}


/* ==========================================================
   MAIN GENERATE BUTTON
   ========================================================== */

#generate-btn {
    margin-top: 18px;

    border: 1px solid #444 !important;

    border-radius: 9px !important;

    background:
        linear-gradient(
            180deg,
            #e9e9e9,
            #cfcfcf
        ) !important;

    color: #080808 !important;

    font-weight: 600 !important;

    transition:
        transform .2s ease,
        box-shadow .2s ease !important;
}

#generate-btn:hover {
    transform: translateY(-2px);

    box-shadow:
        0 10px 30px rgba(255,255,255,.08) !important;
}


/* ==========================================================
   RESPONSE
   ========================================================== */

.response-panel {
    min-height: 360px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.response-status {
    display: inline-flex;

    align-items: center;

    gap: 7px;

    padding: 6px 10px;

    border: 1px solid #303030;

    border-radius: 999px;

    color: #858585;

    font-size: 9px;

    text-transform: uppercase;
    letter-spacing: .1em;

    align-self: flex-start;
}

.response-status.clear {
    color: #8ce0d2;

    border-color: rgba(140,224,210,.24);

    background: rgba(140,224,210,.07);
}

.response-status.flagged {
    color: #e4bd7d;

    border-color: rgba(228,189,125,.25);

    background: rgba(228,189,125,.07);
}

.response-content {
    margin-top: 20px;

    color: #dddddd;

    font-size: 13px;

    line-height: 1.8;

    white-space: pre-wrap;
}

.guardrail-note {
    margin-top: 17px;

    padding: 12px 14px;

    border-left: 2px solid #e4bd7d;

    background: rgba(228,189,125,.045);

    color: #999999;

    font-size: 10px;

    line-height: 1.65;
}


/* ==========================================================
   METRIC CARDS
   ========================================================== */

.metric-grid {
    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 12px;
}

.metric-card {
    padding: 19px;

    border: 1px solid #292929;

    border-radius: 13px;

    background: #101010;

    transition:
        transform .22s ease,
        border-color .22s ease,
        background .22s ease,
        box-shadow .22s ease;
}

.metric-card:hover {
    transform: translateY(-4px);

    border-color: #414141;

    background: #141414;

    box-shadow:
        0 16px 40px rgba(0,0,0,.2);
}

.metric-label {
    margin-bottom: 14px;

    color: #696969;

    font-size: 9px;

    text-transform: uppercase;
    letter-spacing: .15em;
}

.metric-value {
    color: #f0f0f0;

    font-family: "Space Grotesk", sans-serif;

    font-size: 30px;

    letter-spacing: -.045em;
}

.metric-baseline {
    margin-top: 7px;

    color: #777777;

    font-size: 10px;
}

.metric-delta {
    margin-top: 10px;

    color: #8ce0d2;

    font-size: 10px;

    font-weight: 600;
}

.metric-delta.warn {
    color: #e4bd7d;
}

.metric-ci {
    margin-top: 6px;

    color: #555555;

    font-size: 9px;
}


/* ==========================================================
   EVALUATION TABLE
   ========================================================== */

.eval-table-wrap {
    margin-top: 19px;

    overflow-x: auto;

    border: 1px solid #292929;

    border-radius: 13px;
}

.eval-table {
    width: 100%;

    min-width: 720px;

    border-collapse: collapse;
}

.eval-table th {
    padding: 13px 15px;

    background: #151515;

    color: #666666;

    font-size: 9px;

    text-align: left;

    text-transform: uppercase;
    letter-spacing: .13em;
}

.eval-table td {
    padding: 13px 15px;

    border-top: 1px solid #202020;

    color: #bdbdbd;

    font-size: 11px;
}

.eval-table tr:hover td {
    background: #121212;
}

.positive {
    color: #8ce0d2 !important;
}

.negative {
    color: #e4bd7d !important;
}


/* ==========================================================
   PROTOCOL STRIP
   ========================================================== */

.protocol {
    display: grid;

    grid-template-columns:
        repeat(5, 1fr);

    margin-top: 19px;

    overflow: hidden;

    border: 1px solid #292929;

    border-radius: 13px;
}

.protocol-item {
    padding: 15px;

    border-right: 1px solid #292929;

    background: #101010;
}

.protocol-item:last-child {
    border-right: none;
}

.protocol-label {
    color: #666666;

    font-size: 8px;

    text-transform: uppercase;
    letter-spacing: .13em;
}

.protocol-value {
    margin-top: 7px;

    color: #d0d0d0;

    font-size: 10px;
}

.engineering-note {
    margin-top: 13px;

    padding: 13px 15px;

    border: 1px solid #282828;

    border-radius: 11px;

    background: #0e0e0e;

    color: #888888;

    font-size: 10px;

    line-height: 1.7;
}

.engineering-note strong {
    color: #bcbcbc;
}


/* ==========================================================
   PIPELINE
   ========================================================== */

.pipeline {
    display: grid;

    grid-template-columns:
        repeat(5, 1fr);

    gap: 12px;
}

.pipeline-step {
    min-height: 145px;

    padding: 19px;

    border: 1px solid #292929;

    border-radius: 13px;

    background: #101010;

    transition:
        transform .22s ease,
        border-color .22s ease,
        background .22s ease;
}

.pipeline-step:hover {
    transform: translateY(-4px);

    border-color: #404040;

    background: #141414;
}

.pipeline-dot {
    width: 8px;
    height: 8px;

    margin-bottom: 17px;

    border-radius: 50%;

    background: #8ce0d2;

    box-shadow:
        0 0 12px rgba(140,224,210,.35);
}

.pipeline-title {
    color: #ededed;

    font-family: "Space Grotesk", sans-serif;

    font-size: 14px;

    font-weight: 600;
}

.pipeline-copy {
    margin-top: 8px;

    color: #777777;

    font-size: 10px;

    line-height: 1.65;
}


/* ==========================================================
   CONFIG
   ========================================================== */

.config-grid {
    display: grid;

    grid-template-columns:
        repeat(2, 1fr);

    gap: 13px;
}

.config-card {
    padding: 19px;

    border: 1px solid #292929;

    border-radius: 13px;

    background: #101010;

    transition:
        transform .2s ease,
        border-color .2s ease;
}

.config-card:hover {
    transform: translateY(-2px);

    border-color: #3b3b3b;
}

.config-heading {
    margin-bottom: 13px;

    color: #ededed;

    font-family: "Space Grotesk", sans-serif;

    font-size: 14px;
}

.config-row {
    display: flex;

    justify-content: space-between;

    gap: 20px;

    padding: 8px 0;

    border-bottom: 1px solid #202020;

    font-size: 10px;
}

.config-row:last-child {
    border-bottom: none;
}

.config-key {
    color: #666666;
}

.config-value {
    color: #c9c9c9;

    text-align: right;
}


/* ==========================================================
   GUARDRAIL
   ========================================================== */

.guardrail-grid {
    display: grid;

    grid-template-columns:
        1.1fr .9fr;

    gap: 14px;
}

.guardrail-card {
    padding: 21px;

    border: 1px solid #292929;

    border-radius: 13px;

    background: #101010;

    transition:
        border-color .2s ease,
        transform .2s ease;
}

.guardrail-card:hover {
    transform: translateY(-2px);

    border-color: #3b3b3b;
}

.guardrail-title {
    margin-bottom: 10px;

    color: #ededed;

    font-family: "Space Grotesk", sans-serif;

    font-size: 16px;
}

.guardrail-copy {
    color: #808080;

    font-size: 11px;

    line-height: 1.75;
}

.tag-row {
    display: flex;

    flex-wrap: wrap;

    gap: 7px;

    margin-top: 17px;
}

.tag {
    padding: 6px 9px;

    border: 1px solid #303030;

    border-radius: 999px;

    color: #929292;

    font-size: 8px;

    text-transform: uppercase;

    letter-spacing: .07em;
}

.guardrail-flow {
    display: flex;

    align-items: center;

    flex-wrap: wrap;

    gap: 9px;

    margin-top: 20px;

    color: #c0c0c0;

    font-size: 10px;
}

.flow-arrow {
    color: #555555;
}

.guardrail-stat {
    display: flex;

    justify-content: space-between;

    padding: 11px 0;

    border-bottom: 1px solid #202020;

    font-size: 10px;
}

.guardrail-stat:last-child {
    border-bottom: none;
}

.guardrail-stat span:first-child {
    color: #656565;
}

.guardrail-stat span:last-child {
    color: #c5c5c5;
}


/* ==========================================================
   FACTS
   ========================================================== */

.facts-grid {
    display: grid;

    grid-template-columns:
        repeat(4, 1fr);

    gap: 14px;
}

.fact {
    padding: 18px 19px;

    border: 1px solid #292929;

    border-radius: 13px;

    background: #101010;

    transition:
        transform .2s ease,
        border-color .2s ease;
}

.fact:hover {
    transform: translateY(-2px);

    border-color: #3b3b3b;
}

.fact-label {
    color: #656565;

    font-size: 8px;

    text-transform: uppercase;

    letter-spacing: .14em;
}

.fact-value {
    margin-top: 8px;

    color: #eeeeee;

    font-family: "Space Grotesk", sans-serif;

    font-size: 20px;

    letter-spacing: -.025em;
}

.fact-description {
    margin-top: 5px;

    color: #747474;

    font-size: 10px;

    line-height: 1.6;
}


/* ==========================================================
   FOOTER
   ========================================================== */

.footer {
    display: flex;

    justify-content: space-between;

    margin-top: 78px;

    padding: 24px 0 35px;

    border-top: 1px solid #252525;

    color: #5d5d5d;

    font-size: 8px;

    text-transform: uppercase;

    letter-spacing: .13em;
}


/* ==========================================================
   RESPONSIVE
   ========================================================== */

@media (max-width: 1150px) {

    .nav-links {
        display: none;
    }

    .hero-grid {
        grid-template-columns: 1fr;
    }

    .hero-readout {
        max-width: 500px;
    }

    .metric-grid {
        grid-template-columns:
            repeat(2, 1fr);
    }

    .pipeline {
        grid-template-columns:
            repeat(2, 1fr);
    }

    .facts-grid {
        grid-template-columns:
            repeat(2, 1fr);
    }
}


@media (max-width: 850px) {

    .workspace-grid,
    .guardrail-grid,
    .config-grid {
        grid-template-columns: 1fr;
    }

    .protocol {
        grid-template-columns:
            repeat(2, 1fr);
    }

    .protocol-item {
        border-bottom: 1px solid #292929;
    }
}


@media (max-width: 600px) {

    .top-nav {
        padding: 13px 5px;
    }

    .nav-meta {
        display: none;
    }

    .hero {
        padding-top: 45px;
    }

    .hero-title {
        font-size: 45px;
    }

    .metric-grid,
    .pipeline,
    .facts-grid,
    .protocol {
        grid-template-columns: 1fr;
    }

    .protocol-item {
        border-right: none;
    }

    .footer {
        flex-direction: column;

        gap: 9px;
    }
}

"""


# ============================================================
# NAVIGATION
# ============================================================

def nav_html():
    return """
    <div class="top-nav">

        <div class="brand">
            <div class="brand-mark">NX</div>
            <div class="brand-name">NexaTune AI</div>
        </div>

        <div class="nav-links">

            <a class="nav-link" href="#workspace">
                Workspace
            </a>

            <a class="nav-link" href="#evaluation">
                Evaluation
            </a>

            <a class="nav-link" href="#pipeline">
                Pipeline
            </a>

            <a class="nav-link" href="#configuration">
                Configuration
            </a>

            <a class="nav-link" href="#guardrail">
                Guardrail
            </a>

            <a class="nav-link" href="#model">
                Model
            </a>

        </div>

        <div class="nav-meta">
            <span class="status-dot"></span>
            <span>Local inference</span>
        </div>

    </div>
    """


# ============================================================
# HERO
# ============================================================

def hero_html():
    return """
    <div class="hero">

        <div class="hero-grid">

            <div>

                <div class="kicker">
                    DOMAIN-ADAPTIVE FINE-TUNING
                </div>

                <h1 class="hero-title">
                    A support model<br>
                    that knows its limits.
                </h1>

                <div class="hero-copy">
                    NexaTune adapts Qwen3-1.7B to customer-support
                    conversations using QLoRA-based supervised fine-tuning,
                    evaluates the tuned model against the untouched base model,
                    and serves it behind a capability-aware guardrail.
                </div>

            </div>


            <div class="hero-readout">

                <div class="readout-label">
                    Engineering readout
                </div>

                <div class="readout-main">
                    LoRA 16 / 32
                </div>

                <div class="readout-sub">
                    6.42M trainable parameters · 0.37% of base model<br>
                    21,500 training examples · max sequence length 512<br>
                    ROUGE-L 0.212 → 0.408<br>
                    AI-assisted review · n=30
                </div>

            </div>

        </div>

    </div>
    """


# ============================================================
# WORKSPACE
# ============================================================

def workspace_html():
    return """
    <div
        id="workspace"
        class="section"
        style="margin-top: 0;"
    >

        <div class="section-header">

            <div class="section-title">
                Inference workspace
            </div>

            <div class="section-label">
                Live local model
            </div>

        </div>

    </div>
    """


# ============================================================
# EVALUATION
# ============================================================

def evaluation_html():
    return f"""
    <div
        id="evaluation"
        class="section"
    >

        <div class="section-header">

            <div class="section-title">
                Evaluation
            </div>

            <div class="section-label">
                Base vs NexaTune
            </div>

        </div>


        <div class="metric-grid">

            <div class="metric-card">

                <div class="metric-label">
                    ROUGE-L F1
                </div>

                <div class="metric-value">
                    {NEXA_ROUGE:.4f}
                </div>

                <div class="metric-baseline">
                    Base {BASE_ROUGE:.4f}
                </div>

                <div class="metric-delta">
                    +{ROUGE_DELTA:.4f}
                </div>

                <div class="metric-ci">
                    95% CI {ROUGE_CI}
                </div>

            </div>


            <div class="metric-card">

                <div class="metric-label">
                    BERTScore F1
                </div>

                <div class="metric-value">
                    {NEXA_BERT:.4f}
                </div>

                <div class="metric-baseline">
                    Base {BASE_BERT:.4f}
                </div>

                <div class="metric-delta">
                    +{BERT_DELTA:.4f}
                </div>

                <div class="metric-ci">
                    95% CI {BERT_CI}
                </div>

            </div>


            <div class="metric-card">

                <div class="metric-label">
                    Average latency
                </div>

                <div class="metric-value">
                    {NEXA_LATENCY:.2f}s
                </div>

                <div class="metric-baseline">
                    Base {BASE_LATENCY:.2f}s
                </div>

                <div class="metric-delta warn">
                    +67.6%
                </div>

                <div class="metric-ci">
                    Local inference measurement
                </div>

            </div>


            <div class="metric-card">

                <div class="metric-label">
                    AI-assisted review
                </div>

                <div class="metric-value">
                    +{AI_MARGIN:.2f}
                </div>

                <div class="metric-baseline">
                    Overall mean margin
                </div>

                <div class="metric-delta">
                    NexaTune − Base
                </div>

                <div class="metric-ci">
                    95% CI {AI_CI}
                </div>

            </div>

        </div>


        <div class="eval-table-wrap">

            <table class="eval-table">

                <thead>

                    <tr>
                        <th>AI-assisted review · n=30</th>
                        <th>Base</th>
                        <th>NexaTune</th>
                        <th>Δ</th>
                    </tr>

                </thead>


                <tbody>

                    <tr>
                        <td>Relevance</td>
                        <td>4.10</td>
                        <td>4.57</td>
                        <td class="positive">+0.47</td>
                    </tr>

                    <tr>
                        <td>Helpfulness</td>
                        <td>3.43</td>
                        <td>3.97</td>
                        <td class="positive">+0.53</td>
                    </tr>

                    <tr>
                        <td>Correctness</td>
                        <td>3.50</td>
                        <td>3.83</td>
                        <td class="positive">+0.33</td>
                    </tr>

                    <tr>
                        <td>Professionalism</td>
                        <td>4.30</td>
                        <td>4.60</td>
                        <td class="positive">+0.30</td>
                    </tr>

                    <tr>
                        <td>Factuality</td>
                        <td>4.37</td>
                        <td>3.33</td>
                        <td class="negative">−1.03</td>
                    </tr>

                </tbody>

            </table>

        </div>


        <div class="protocol">

            <div class="protocol-item">

                <div class="protocol-label">
                    Benchmark
                </div>

                <div class="protocol-value">
                    100 held-out examples
                </div>

            </div>


            <div class="protocol-item">

                <div class="protocol-label">
                    Decoding
                </div>

                <div class="protocol-value">
                    Deterministic
                </div>

            </div>


            <div class="protocol-item">

                <div class="protocol-label">
                    Max tokens
                </div>

                <div class="protocol-value">
                    128
                </div>

            </div>


            <div class="protocol-item">

                <div class="protocol-label">
                    Output length
                </div>

                <div class="protocol-value">
                    100.1 → 99.9 tokens
                </div>

            </div>


            <div class="protocol-item">

                <div class="protocol-label">
                    Bootstrap
                </div>

                <div class="protocol-value">
                    10,000 resamples
                </div>

            </div>

        </div>


        <div class="engineering-note">

            <strong>Engineering trade-off ·</strong>

            Fine-tuning improved benchmark similarity and semantic quality
            while increasing measured local inference latency.

            Output length remained essentially unchanged
            (100.1 → 99.9 tokens), so the quality change was not driven by
            substantially longer generations.

        </div>

    </div>
    """


# ============================================================
# PIPELINE
# ============================================================

def pipeline_html():
    return """
    <div
        id="pipeline"
        class="section"
    >

        <div class="section-header">

            <div class="section-title">
                Engineering pipeline
            </div>

            <div class="section-label">
                End-to-end lifecycle
            </div>

        </div>


        <div class="pipeline">


            <div class="pipeline-step">

                <div class="pipeline-dot"></div>

                <div class="pipeline-title">
                    Dataset
                </div>

                <div class="pipeline-copy">
                    26,872 original examples<br>
                    21,500 train · 2,713 val · 2,659 test
                </div>

            </div>


            <div class="pipeline-step">

                <div class="pipeline-dot"></div>

                <div class="pipeline-title">
                    QLoRA + SFT
                </div>

                <div class="pipeline-copy">
                    Customer → Assistant<br>
                    4-bit NF4 · LoRA r=16
                </div>

            </div>


            <div class="pipeline-step">

                <div class="pipeline-dot"></div>

                <div class="pipeline-title">
                    Benchmark
                </div>

                <div class="pipeline-copy">
                    Base vs tuned<br>
                    ROUGE-L · BERTScore · latency
                </div>

            </div>


            <div class="pipeline-step">

                <div class="pipeline-dot"></div>

                <div class="pipeline-title">
                    Serving
                </div>

                <div class="pipeline-copy">
                    FastAPI<br>
                    Qwen3-1.7B + adapter
                </div>

            </div>


            <div class="pipeline-step">

                <div class="pipeline-dot"></div>

                <div class="pipeline-title">
                    Guardrail
                </div>

                <div class="pipeline-copy">
                    Detect → Replace<br>
                    Safe response
                </div>

            </div>


        </div>

    </div>
    """


# ============================================================
# CONFIGURATION
# ============================================================

def config_html():
    return """
    <div
        id="configuration"
        class="section"
    >

        <div class="section-header">

            <div class="section-title">
                Model configuration
            </div>

            <div class="section-label">
                Training + deployment
            </div>

        </div>


        <div class="config-grid">


            <div class="config-card">

                <div class="config-heading">
                    Base model
                </div>

                <div class="config-row">
                    <span class="config-key">Model</span>
                    <span class="config-value">
                        Qwen/Qwen3-1.7B-Base
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Parameters</span>
                    <span class="config-value">
                        1.727B
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Quantization</span>
                    <span class="config-value">
                        4-bit NF4
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Double quant</span>
                    <span class="config-value">
                        Enabled
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Compute dtype</span>
                    <span class="config-value">
                        FP16
                    </span>
                </div>

            </div>


            <div class="config-card">

                <div class="config-heading">
                    LoRA adapter
                </div>

                <div class="config-row">
                    <span class="config-key">Rank</span>
                    <span class="config-value">
                        16
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Alpha</span>
                    <span class="config-value">
                        32
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Dropout</span>
                    <span class="config-value">
                        0.05
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Target modules</span>
                    <span class="config-value">
                        q/k/v/o_proj
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Adapter size</span>
                    <span class="config-value">
                        ≈24.5 MB
                    </span>
                </div>

            </div>


            <div class="config-card">

                <div class="config-heading">
                    Training
                </div>

                <div class="config-row">
                    <span class="config-key">Objective</span>
                    <span class="config-value">
                        Supervised fine-tuning
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Epochs</span>
                    <span class="config-value">
                        1
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Learning rate</span>
                    <span class="config-value">
                        2e-4
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Batch size</span>
                    <span class="config-value">
                        1 × 8 accumulation
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Max length</span>
                    <span class="config-value">
                        512 tokens
                    </span>
                </div>

            </div>


            <div class="config-card">

                <div class="config-heading">
                    Parameter efficiency
                </div>

                <div class="config-row">
                    <span class="config-key">Trainable params</span>
                    <span class="config-value">
                        6.42M
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Total params</span>
                    <span class="config-value">
                        1.727B
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Trainable ratio</span>
                    <span class="config-value">
                        0.37%
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Fine-tuning</span>
                    <span class="config-value">
                        QLoRA
                    </span>
                </div>

                <div class="config-row">
                    <span class="config-key">Serving</span>
                    <span class="config-value">
                        FastAPI + adapter
                    </span>
                </div>

            </div>


        </div>

    </div>
    """


# ============================================================
# GUARDRAIL
# ============================================================

def guardrail_html():
    return """
    <div
        id="guardrail"
        class="section"
    >

        <div class="section-header">

            <div class="section-title">
                Responsible AI guardrail
            </div>

            <div class="section-label">
                Capability boundary enforcement
            </div>

        </div>


        <div class="guardrail-grid">


            <div class="guardrail-card">

                <div class="guardrail-title">
                    The model should not claim actions it cannot perform.
                </div>

                <div class="guardrail-copy">
                    NexaTune's generation output is scanned after inference
                    for unsupported operational claims. When a generated
                    response claims an unavailable capability, the system
                    replaces it with a bounded customer-facing response
                    instead of allowing the model to imply that a backend
                    action actually occurred.
                </div>


                <div class="tag-row">

                    <span class="tag">
                        Human handoff
                    </span>

                    <span class="tag">
                        Refund actions
                    </span>

                    <span class="tag">
                        Order cancellation
                    </span>

                    <span class="tag">
                        Backend actions
                    </span>

                    <span class="tag">
                        Unsupported commitments
                    </span>

                </div>


                <div class="guardrail-flow">

                    <span>Generate</span>

                    <span class="flow-arrow">→</span>

                    <span>Detect</span>

                    <span class="flow-arrow">→</span>

                    <span>Replace</span>

                    <span class="flow-arrow">→</span>

                    <span>Return</span>

                </div>

            </div>


            <div class="guardrail-card">

                <div class="guardrail-title">
                    Runtime policy
                </div>


                <div class="guardrail-stat">
                    <span>Detection</span>
                    <span>Pattern scan</span>
                </div>

                <div class="guardrail-stat">
                    <span>Remediation</span>
                    <span>Safe response</span>
                </div>

                <div class="guardrail-stat">
                    <span>Traceability</span>
                    <span>Original output retained</span>
                </div>

                <div class="guardrail-stat">
                    <span>API status</span>
                    <span>Bounded capability</span>
                </div>

            </div>


        </div>

    </div>
    """


# ============================================================
# MODEL SNAPSHOT
# ============================================================

def facts_html():
    return """
    <div
        id="model"
        class="section"
    >

        <div class="section-header">

            <div class="section-title">
                Model snapshot
            </div>

            <div class="section-label">
                Project facts
            </div>

        </div>


        <div class="facts-grid">


            <div class="fact">

                <div class="fact-label">
                    BASE MODEL
                </div>

                <div class="fact-value">
                    Qwen3-1.7B
                </div>

                <div class="fact-description">
                    1.727B total parameters
                </div>

            </div>


            <div class="fact">

                <div class="fact-label">
                    DATASET
                </div>

                <div class="fact-value">
                    26,872
                </div>

                <div class="fact-description">
                    21,500 train · 2,713 val · 2,659 test
                </div>

            </div>


            <div class="fact">

                <div class="fact-label">
                    ADAPTER
                </div>

                <div class="fact-value">
                    24.5 MB
                </div>

                <div class="fact-description">
                    LoRA adapter artifact
                </div>

            </div>


            <div class="fact">

                <div class="fact-label">
                    SERVING
                </div>

                <div class="fact-value">
                    FastAPI
                </div>

                <div class="fact-description">
                    QLoRA + guardrail inference
                </div>

            </div>


        </div>

    </div>
    """


# ============================================================
# FOOTER
# ============================================================

def footer_html():
    return """
    <div class="footer">

        <span>
            NexaTune AI
        </span>

        <span>
            Python · QLoRA · PEFT · FastAPI · Gradio
        </span>

    </div>
    """


# ============================================================
# API HEALTH
# ============================================================

def check_api():
    try:

        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if response.status_code == 200:

            data = response.json()

            if data.get("model_loaded"):
                return "ONLINE · MODEL LOADED"

            return "ONLINE · MODEL NOT LOADED"

        return f"API ERROR · {response.status_code}"

    except requests.exceptions.ConnectionError:

        return "SYSTEM OFFLINE · START FASTAPI"

    except Exception:

        return "SYSTEM CHECK FAILED"


# ============================================================
# INFERENCE
# ============================================================

def generate_response(message):

    if not message or not message.strip():

        return """
        <div class="response-status">

            <span>●</span>

            Awaiting input

        </div>

        <div class="response-content">

            Enter a customer-support message to run NexaTune.

        </div>
        """


    try:

        response = requests.post(

            f"{API_URL}/generate",

            json={
                "message": message.strip()
            },

            timeout=120,

        )


        # ----------------------------------------------------
        # VALIDATION ERROR
        # ----------------------------------------------------

        if response.status_code == 422:

            try:

                data = response.json()

                detail = data.get(
                    "detail",
                    "Invalid request.",
                )

            except Exception:

                detail = "Invalid request."


            return f"""
            <div class="response-status flagged">

                <span>●</span>

                REQUEST REJECTED

            </div>

            <div class="response-content">

                {html.escape(str(detail))}

            </div>
            """


        # ----------------------------------------------------
        # MODEL UNAVAILABLE
        # ----------------------------------------------------

        if response.status_code == 503:

            return """
            <div class="response-status flagged">

                <span>●</span>

                MODEL UNAVAILABLE

            </div>

            <div class="response-content">

                The inference model is not currently available.

            </div>
            """


        response.raise_for_status()


        data = response.json()


        generated = data.get(
            "response",
            "No response returned.",
        )


        flagged = data.get(
            "guardrail_flagged",
            False,
        )


        categories = data.get(
            "guardrail_categories",
            [],
        )


        # ----------------------------------------------------
        # GUARDRAIL TRIGGERED
        # ----------------------------------------------------

        if flagged:

            category_text = ", ".join(

                category.replace(
                    "_",
                    " ",
                ).title()

                for category in categories

            )


            return f"""
            <div class="response-status flagged">

                <span>●</span>

                GUARDRAIL REMEDIATED

            </div>


            <div class="response-content">

                {html.escape(generated)}

            </div>


            <div class="guardrail-note">

                Capability guardrail activated.

                Category:
                {html.escape(
                    category_text
                    or "unsupported capability"
                )}

                <br>

                The original model output was replaced with a
                bounded customer-facing response.

            </div>
            """


        # ----------------------------------------------------
        # NORMAL RESPONSE
        # ----------------------------------------------------

        return f"""
        <div class="response-status clear">

            <span>●</span>

            GUARDRAIL CLEAR

        </div>


        <div class="response-content">

            {html.escape(generated)}

        </div>
        """


    except requests.exceptions.ConnectionError:

        return """
        <div class="response-status flagged">

            <span>●</span>

            SYSTEM OFFLINE

        </div>


        <div class="response-content">

            Start FastAPI on 127.0.0.1:8000 before running inference.

        </div>
        """


    except requests.exceptions.Timeout:

        return """
        <div class="response-status flagged">

            <span>●</span>

            INFERENCE TIMEOUT

        </div>


        <div class="response-content">

            The model took too long to respond.
            Check the FastAPI terminal for inference logs.

        </div>
        """


    except Exception as exc:

        return f"""
        <div class="response-status flagged">

            <span>●</span>

            REQUEST FAILED

        </div>


        <div class="response-content">

            {html.escape(str(exc))}

        </div>
        """


# ============================================================
# QUICK PROMPTS
# ============================================================

PROMPT_INVOICE = (
    "Where can I download my invoice for order #00108?"
)

PROMPT_SHIPPING = (
    "My package has not arrived yet. "
    "Can you help me check the shipping status?"
)

PROMPT_REFUND = (
    "Please process my refund immediately."
)

PROMPT_ACCOUNT = (
    "I cannot access my account and need help resetting my password."
)

PROMPT_HUMAN = (
    "I need to speak to a human representative."
)


# ============================================================
# GRADIO APPLICATION
# ============================================================

with gr.Blocks(
    title="NexaTune AI",
    css=CSS,

    theme=gr.themes.Base(

        primary_hue="neutral",
        secondary_hue="neutral",
        neutral_hue="neutral",

        font=[
            "Inter",
            "sans-serif",
        ],

        font_mono=[
            "JetBrains Mono",
            "monospace",
        ],
    ),

) as demo:


    # ========================================================
    # NAV
    # ========================================================

    gr.HTML(nav_html())


    # ========================================================
    # HERO
    # ========================================================

    gr.HTML(hero_html())


    # ========================================================
    # WORKSPACE HEADER
    # ========================================================

    gr.HTML(workspace_html())


    # ========================================================
    # LIVE WORKSPACE
    # ========================================================

    with gr.Row(
        elem_classes="workspace",
    ):

        with gr.Column(
            scale=1,
            elem_classes="workspace-panel",
        ):

            gr.HTML(
                """
                <div class="panel-label">
                    Customer message
                </div>
                """
            )


            message_input = gr.Textbox(

                placeholder=(
                    "Describe the customer's issue..."
                ),

                lines=10,

                max_lines=14,

                show_label=False,

                container=False,

            )


            gr.HTML(
                """
                <div
                    class="panel-label"
                    style="margin-top:16px;"
                >
                    Quick scenarios
                </div>
                """
            )


            with gr.Row(
                elem_classes="quick-row",
            ):

                invoice_btn = gr.Button(
                    "Invoice",
                    size="sm",
                    elem_classes="quick-btn",
                )

                shipping_btn = gr.Button(
                    "Shipping",
                    size="sm",
                    elem_classes="quick-btn",
                )

                refund_btn = gr.Button(
                    "Refund",
                    size="sm",
                    elem_classes="quick-btn",
                )

                account_btn = gr.Button(
                    "Account",
                    size="sm",
                    elem_classes="quick-btn",
                )

                human_btn = gr.Button(
                    "Human",
                    size="sm",
                    elem_classes="quick-btn",
                )


            generate_btn = gr.Button(

                "Run NexaTune Inference",

                variant="primary",

                size="lg",

                elem_id="generate-btn",

            )


        with gr.Column(
            scale=1,
            elem_classes=[
                "workspace-panel",
                "response-panel",
            ],
        ):

            gr.HTML(
                """
                <div class="panel-label">
                    Model response
                </div>
                """
            )


            response_output = gr.HTML(

                """
                <div class="response-status">

                    <span>●</span>

                    Awaiting input

                </div>


                <div class="response-content">

                    Enter a customer-support message
                    to run NexaTune.

                </div>
                """

            )


    # ========================================================
    # EVALUATION
    # ========================================================

    gr.HTML(
        evaluation_html()
    )


    # ========================================================
    # PIPELINE
    # ========================================================

    gr.HTML(
        pipeline_html()
    )


    # ========================================================
    # CONFIGURATION
    # ========================================================

    gr.HTML(
        config_html()
    )


    # ========================================================
    # GUARDRAIL
    # ========================================================

    gr.HTML(
        guardrail_html()
    )


    # ========================================================
    # MODEL
    # ========================================================

    gr.HTML(
        facts_html()
    )


    # ========================================================
    # FOOTER
    # ========================================================

    gr.HTML(
        footer_html()
    )


    # ========================================================
    # EVENTS
    # ========================================================

    generate_btn.click(

        fn=generate_response,

        inputs=message_input,

        outputs=response_output,

    )


    invoice_btn.click(

        fn=lambda: PROMPT_INVOICE,

        inputs=None,

        outputs=message_input,

    )


    shipping_btn.click(

        fn=lambda: PROMPT_SHIPPING,

        inputs=None,

        outputs=message_input,

    )


    refund_btn.click(

        fn=lambda: PROMPT_REFUND,

        inputs=None,

        outputs=message_input,

    )


    account_btn.click(

        fn=lambda: PROMPT_ACCOUNT,

        inputs=None,

        outputs=message_input,

    )


    human_btn.click(

        fn=lambda: PROMPT_HUMAN,

        inputs=None,

        outputs=message_input,

    )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":

    demo.launch(

        server_name="127.0.0.1",

        server_port=7860,

        show_error=True,

    )