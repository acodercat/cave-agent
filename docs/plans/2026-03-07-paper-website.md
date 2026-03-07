# CaveAgent Paper Website Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an interactive academic project website for the CaveAgent paper using the Academic Project Page Template with enhanced interactions.

**Architecture:** Static HTML site based on the Academic Project Page Template (Bulma CSS). Custom CSS and vanilla JS add interactive elements: a dual-stream step-by-step animation, a smart home walkthrough with animated device cards, and tabbed experiment results. All assets are static; PDF figures are pre-converted to PNG.

**Tech Stack:** HTML5, Bulma CSS, Prism.js (code highlighting), vanilla JS, GitHub Pages deployment

---

### Task 1: Scaffold — Clone template assets and create directory structure

**Files:**
- Create: `Website/index.html` (empty placeholder)
- Create: `Website/static/css/caveagent.css` (empty)
- Create: `Website/static/js/caveagent.js` (empty)
- Create: `Website/.nojekyll`

**Step 1: Download template static assets into Website/static/**

```bash
cd /mnt/data/wanzl/cave-agent
mkdir -p Website/static/{css,js,images,images/institution_logos}

# Download Bulma and template CSS/JS from the Academic Project Page Template repo
curl -sL "https://raw.githubusercontent.com/eliahuhorwitz/Academic-project-page-template/master/static/css/bulma.min.css" -o Website/static/css/bulma.min.css
curl -sL "https://raw.githubusercontent.com/eliahuhorwitz/Academic-project-page-template/master/static/css/bulma-carousel.min.css" -o Website/static/css/bulma-carousel.min.css
curl -sL "https://raw.githubusercontent.com/eliahuhorwitz/Academic-project-page-template/master/static/css/bulma-slider.min.css" -o Website/static/css/bulma-slider.min.css
curl -sL "https://raw.githubusercontent.com/eliahuhorwitz/Academic-project-page-template/master/static/css/index.css" -o Website/static/css/index.css
curl -sL "https://raw.githubusercontent.com/eliahuhorwitz/Academic-project-page-template/master/static/css/fontawesome.all.min.css" -o Website/static/css/fontawesome.all.min.css
curl -sL "https://raw.githubusercontent.com/eliahuhorwitz/Academic-project-page-template/master/static/js/index.js" -o Website/static/js/index.js
curl -sL "https://raw.githubusercontent.com/eliahuhorwitz/Academic-project-page-template/master/static/js/bulma-carousel.min.js" -o Website/static/js/bulma-carousel.min.js
curl -sL "https://raw.githubusercontent.com/eliahuhorwitz/Academic-project-page-template/master/static/js/bulma-slider.min.js" -o Website/static/js/bulma-slider.min.js
curl -sL "https://raw.githubusercontent.com/eliahuhorwitz/Academic-project-page-template/master/static/js/fontawesome.all.min.js" -o Website/static/js/fontawesome.all.min.js

# Download Prism.js (Python + copy-to-clipboard)
curl -sL "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js" -o Website/static/js/prism.min.js
curl -sL "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js" -o Website/static/js/prism-python.min.js
curl -sL "https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" -o Website/static/css/prism-tomorrow.min.css

# Create empty custom files
touch Website/static/css/caveagent.css
touch Website/static/js/caveagent.js
touch Website/.nojekyll
```

**Step 2: Copy images from paper figures into Website/static/images/**

```bash
cd /mnt/data/wanzl/cave-agent

# Copy PNG figures
cp Website/Cave_agent_overleaf_src/figures/CaveAgent/Logo-CaveAgent.png Website/static/images/logo.png
cp Website/Cave_agent_overleaf_src/figures/CaveAgent/framework.png Website/static/images/framework.png
cp Website/Cave_agent_overleaf_src/figures/CaveAgent/smarthome.png Website/static/images/smarthome.png
cp Website/Cave_agent_overleaf_src/figures/CaveAgent/town_simulation.png Website/static/images/town_simulation.png
cp Website/Cave_agent_overleaf_src/figures/CaveAgent/downstream_applications.png Website/static/images/downstream_applications.png
cp Website/Cave_agent_overleaf_src/figures/CaveAgent/v2_new_elements/Skills.png Website/static/images/skills.png
cp Website/Cave_agent_overleaf_src/figures/CaveAgent/v2_new_elements/AutoML.png Website/static/images/automl.png
cp Website/Cave_agent_overleaf_src/figures/CaveAgent/v2_new_elements/Geospacial_Analysis.png Website/static/images/geospatial.png
cp Website/Cave_agent_overleaf_src/figures/CaveAgent/evolve.png Website/static/images/evolve.png

# Copy institution logos
cp Website/Cave_agent_overleaf_src/figures/hkbu_logo.png Website/static/images/institution_logos/
cp Website/Cave_agent_overleaf_src/figures/hkgai_logo.png Website/static/images/institution_logos/
cp Website/Cave_agent_overleaf_src/figures/hku_logo.png Website/static/images/institution_logos/
cp Website/Cave_agent_overleaf_src/figures/ntu_logo_color.png Website/static/images/institution_logos/
cp Website/Cave_agent_overleaf_src/figures/nus_logo.png Website/static/images/institution_logos/

# Convert PDF figures to PNG (requires pdftoppm or similar)
# If pdftoppm is available:
for pdf in token_consumption steps_success multi_turn_bar type_proficiency_multi_radar benchmark_combined vars_count_heatmap; do
  pdftoppm -png -r 300 -singlefile \
    "Website/Cave_agent_overleaf_src/figures/CaveAgent/${pdf}.pdf" \
    "Website/static/images/${pdf}" 2>/dev/null || echo "SKIP: ${pdf}.pdf (pdftoppm not available, convert manually)"
done
```

Note: If `pdftoppm` is not available, use `convert` (ImageMagick) or convert PDFs manually. These PNG files are required for Task 6 (Experimental Results).

**Step 3: Verify directory structure**

```bash
find Website/static -type f | head -40
ls Website/static/images/
```

Expected: All CSS/JS files present, all PNG images copied, PDF-converted PNGs present (or noted for manual conversion).

**Step 4: Commit**

```bash
git add Website/static Website/.nojekyll
git commit -m "scaffold: add template assets and paper figures for website"
```

---

### Task 2: Build index.html — Hero + Teaser + Abstract (Sections 1-2)

**Files:**
- Create: `Website/index.html`

**Step 1: Create index.html with full head, Hero section, Teaser, and Abstract**

Write `Website/index.html` with the following structure. This is the largest single file — it contains all 8 sections. We build sections 1-2 now and add the rest in subsequent tasks.

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="title" content="CaveAgent: Transforming LLMs into Stateful Runtime Operators">
  <meta name="description" content="CaveAgent introduces Stateful Runtime Management for LLM agents, achieving +10.5% success rate and 28.4% token reduction through persistent Python runtime with direct variable injection and retrieval.">
  <meta name="keywords" content="LLM, agent, function calling, stateful runtime, code generation, tool use, CaveAgent">
  <meta name="author" content="Maohao Ran, Zhenglin Wan, Cooper Lin, Yanting Zhang">
  <meta property="og:type" content="article">
  <meta property="og:title" content="CaveAgent: Transforming LLMs into Stateful Runtime Operators">
  <meta property="og:description" content="CaveAgent shifts LLM tool use from text-generator to runtime-operator with persistent Python environments.">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="citation_title" content="CaveAgent: Transforming LLMs into Stateful Runtime Operators">
  <meta name="citation_author" content="Ran, Maohao">
  <meta name="citation_author" content="Wan, Zhenglin">
  <meta name="citation_publication_date" content="2026">
  <meta name="citation_pdf_url" content="https://arxiv.org/pdf/2601.01569.pdf">

  <title>CaveAgent: Transforming LLMs into Stateful Runtime Operators</title>

  <link rel="icon" type="image/png" href="static/images/logo.png">
  <link rel="stylesheet" href="static/css/bulma.min.css">
  <link rel="stylesheet" href="static/css/index.css">
  <link rel="stylesheet" href="static/css/caveagent.css">
  <link rel="preload" href="static/css/bulma-carousel.min.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
  <link rel="preload" href="static/css/bulma-slider.min.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
  <link rel="preload" href="static/css/fontawesome.all.min.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
  <link rel="preload" href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css" as="style" onload="this.onload=null;this.rel='stylesheet'">
  <link rel="stylesheet" href="static/css/prism-tomorrow.min.css">
  <noscript>
    <link rel="stylesheet" href="static/css/bulma-carousel.min.css">
    <link rel="stylesheet" href="static/css/bulma-slider.min.css">
    <link rel="stylesheet" href="static/css/fontawesome.all.min.css">
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/jpswalsh/academicons@1/css/academicons.min.css">
  </noscript>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">

  <script defer src="https://ajax.googleapis.com/ajax/libs/jquery/3.5.1/jquery.min.js"></script>
  <script defer src="static/js/fontawesome.all.min.js"></script>
  <script defer src="static/js/bulma-carousel.min.js"></script>
  <script defer src="static/js/bulma-slider.min.js"></script>
  <script defer src="static/js/prism.min.js"></script>
  <script defer src="static/js/prism-python.min.js"></script>
  <script defer src="static/js/index.js"></script>
  <script defer src="static/js/caveagent.js"></script>
</head>
<body>

<!-- ==================== SECTION 1: HERO ==================== -->
<section class="hero">
  <div class="hero-body">
    <div class="container is-max-desktop">
      <div class="columns is-centered">
        <div class="column has-text-centered">
          <!-- Logo -->
          <div class="publication-logo">
            <img src="static/images/logo.png" alt="CaveAgent Logo" style="max-height: 120px; margin-bottom: 1rem;">
          </div>

          <h1 class="title is-1 publication-title">CaveAgent: Transforming LLMs into Stateful Runtime Operators</h1>

          <p class="subtitle is-4" style="margin-bottom: 1.5rem; color: #666; font-style: italic;">
            "From text-in-text-out to (text&amp;object)-in-(text&amp;object)-out"
          </p>

          <!-- Authors -->
          <div class="is-size-5 publication-authors">
            <span class="author-block">Maohao Ran<sup>*1,2</sup>,</span>
            <span class="author-block">Zhenglin Wan<sup>*1,3</sup>,</span>
            <span class="author-block">Cooper Lin<sup>4</sup>,</span>
            <span class="author-block">Yanting Zhang<sup>1</sup>,</span>
            <span class="author-block">Hongyu Xin<sup>1</sup>,</span>
            <span class="author-block">Hongwei Fan<sup>7</sup>,</span>
            <span class="author-block">Yibo Xu<sup>1</sup>,</span>
            <span class="author-block">Beier Luo<sup>4</sup>,</span>
            <span class="author-block">Yaxin Zhou<sup>5</sup>,</span>
            <span class="author-block">Wangbo Zhao<sup>3</sup>,</span>
            <span class="author-block">Lijie Yang<sup>8</sup>,</span>
            <span class="author-block">Lang Feng<sup>6</sup>,</span>
            <span class="author-block">Fuchao Yang<sup>6</sup>,</span>
            <span class="author-block">Jingxuan Wu<sup>9</sup>,</span>
            <span class="author-block">Yiqiao Huang<sup>10</sup>,</span>
            <span class="author-block">Chendong Ma<sup>2</sup>,</span>
            <span class="author-block">Dailing Jiang<sup>2</sup>,</span>
            <span class="author-block">Jianbo Deng<sup>1</sup>,</span>
            <span class="author-block">Sirui Han<sup>1</sup>,</span>
            <span class="author-block">Yang You<sup>3</sup>,</span>
            <span class="author-block">Bo An<sup>6</sup>,</span>
            <span class="author-block">Yike Guo<sup>1</sup>,</span>
            <span class="author-block">Jun Song<sup>&dagger;1,2</sup></span>
          </div>

          <!-- Affiliations -->
          <div class="is-size-6 publication-authors" style="margin-top: 0.5rem;">
            <span class="author-block"><sup>1</sup>HKUST</span>&nbsp;
            <span class="author-block"><sup>2</sup>HKBU</span>&nbsp;
            <span class="author-block"><sup>3</sup>NUS</span>&nbsp;
            <span class="author-block"><sup>4</sup>HKU</span>&nbsp;
            <span class="author-block"><sup>5</sup>CMU</span>&nbsp;
            <span class="author-block"><sup>6</sup>NTU, Singapore</span>&nbsp;
            <span class="author-block"><sup>7</sup>Imperial College London</span>&nbsp;
            <span class="author-block"><sup>8</sup>Princeton</span>&nbsp;
            <span class="author-block"><sup>9</sup>UNC, Chapel Hill</span>&nbsp;
            <span class="author-block"><sup>10</sup>Harvard</span>
          </div>
          <div class="is-size-7" style="margin-top: 0.3rem; color: #888;">
            <sup>*</sup>Equal Contribution &nbsp; <sup>&dagger;</sup>Corresponding Author
          </div>

          <!-- Institution Logos -->
          <div class="institution-logos" style="margin-top: 1rem; display: flex; justify-content: center; align-items: center; gap: 1rem; flex-wrap: wrap;">
            <img src="static/images/institution_logos/hkgai_logo.png" alt="HKGAI" style="height: 35px;">
            <img src="static/images/institution_logos/hkbu_logo.png" alt="HKBU" style="height: 35px;">
            <img src="static/images/institution_logos/nus_logo.png" alt="NUS" style="height: 35px;">
            <img src="static/images/institution_logos/hku_logo.png" alt="HKU" style="height: 35px;">
            <img src="static/images/institution_logos/ntu_logo_color.png" alt="NTU" style="height: 35px;">
          </div>

          <!-- Links -->
          <div class="column has-text-centered" style="margin-top: 1rem;">
            <div class="publication-links">
              <span class="link-block">
                <a href="https://arxiv.org/pdf/2601.01569.pdf" target="_blank" class="external-link button is-normal is-rounded is-dark">
                  <span class="icon"><i class="fas fa-file-pdf"></i></span>
                  <span>Paper</span>
                </a>
              </span>
              <span class="link-block">
                <a href="https://github.com/acodercat/cave-agent" target="_blank" class="external-link button is-normal is-rounded is-dark">
                  <span class="icon"><i class="fab fa-github"></i></span>
                  <span>Code</span>
                </a>
              </span>
              <span class="link-block">
                <a href="https://arxiv.org/abs/2601.01569" target="_blank" class="external-link button is-normal is-rounded is-dark">
                  <span class="icon"><i class="ai ai-arxiv"></i></span>
                  <span>arXiv</span>
                </a>
              </span>
              <span class="link-block">
                <a href="https://pypi.org/project/cave-agent" target="_blank" class="external-link button is-normal is-rounded is-dark">
                  <span class="icon"><i class="fas fa-cube"></i></span>
                  <span>PyPI</span>
                </a>
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== SECTION 2: TEASER + ABSTRACT ==================== -->
<section class="hero teaser">
  <div class="container is-max-desktop">
    <div class="hero-body">
      <img src="static/images/framework.png" alt="CaveAgent Dual-Stream Architecture" style="width: 100%; border: 1px solid #ddd; border-radius: 4px;">
      <h2 class="subtitle has-text-centered" style="margin-top: 1rem;">
        <strong>CaveAgent Dual-Stream Architecture:</strong> Semantic Stream for lightweight reasoning, Runtime Stream for persistent state &amp; data management.
      </h2>
    </div>
  </div>
</section>

<section class="section hero is-light">
  <div class="container is-max-desktop">
    <div class="columns is-centered has-text-centered">
      <div class="column is-four-fifths">
        <h2 class="title is-3">Abstract</h2>
        <div class="content has-text-justified">
          <p>
            LLM-based agents are increasingly capable of complex task execution, yet current agentic systems remain constrained by text-centric paradigms that struggle with long-horizon tasks due to fragile multi-turn dependencies and context drift. We present <strong>CaveAgent</strong>, a framework that shifts LLM tool use from "LLM-as-Text-Generator" to "LLM-as-Runtime-Operator." CaveAgent introduces a dual-stream architecture: a <em>semantic stream</em> for lightweight reasoning and a <em>runtime stream</em> backed by a persistent Python environment for stateful execution.
          </p>
          <p>
            Rather than treating the LLM's text context as the primary workspace, CaveAgent elevates the persistent runtime as the central locus. Beyond leveraging code generation to resolve interdependent sub-tasks in a single step, CaveAgent introduces <strong>Stateful Runtime Management</strong>: it injects, manipulates, and retrieves complex Python objects (e.g., DataFrames, database connections) that persist across turns. CaveAgent further provides a runtime-integrated skill management system that extends the Agent Skills open standard.
          </p>
          <p>
            Evaluations on Tau<sup>2</sup>-bench and BFCL across six SOTA LLMs demonstrate consistent improvements: <strong>+10.5% success rate</strong> on retail tasks, <strong>28.4% reduction</strong> in total token consumption, and <strong>59% token reduction</strong> on data-intensive tasks. The accessible runtime state further provides programmatically verifiable feedback, enabling automated evaluation and reward signal generation for Reinforcement Learning with Verifiable Rewards (RLVR).
          </p>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- Remaining sections added in Tasks 3-7 -->

</body>
</html>
```

**Step 2: Open in browser to verify Hero + Abstract render correctly**

```bash
# Open local file in browser (or use python http server)
cd /mnt/data/wanzl/cave-agent/Website
python3 -m http.server 8080 &
# Then visit http://localhost:8080
```

Expected: Logo, title, authors, institution logos, link buttons, teaser image, and abstract all render. Bulma grid layout is responsive.

**Step 3: Commit**

```bash
git add Website/index.html
git commit -m "feat(website): add Hero, Teaser, and Abstract sections"
```

---

### Task 3: Add Section 3 — Interactive Dual-Stream Animation

**Files:**
- Modify: `Website/index.html` (insert after abstract section)
- Modify: `Website/static/css/caveagent.css`
- Modify: `Website/static/js/caveagent.js`

**Step 1: Add HTML for dual-stream section in index.html**

Insert after the abstract `</section>` closing tag:

```html
<!-- ==================== SECTION 3: HOW CAVEAGENT WORKS ==================== -->
<section class="section">
  <div class="container is-max-desktop">
    <h2 class="title is-3 has-text-centered">How CaveAgent Works</h2>
    <p class="subtitle has-text-centered">Click each step or press "Play All" to see the dual-stream architecture in action.</p>

    <!-- Step Controls -->
    <div class="has-text-centered" style="margin-bottom: 2rem;">
      <button class="button is-rounded is-info" onclick="dualStreamPlay()" id="ds-play-btn">
        <span class="icon"><i class="fas fa-play"></i></span><span>Play All</span>
      </button>
      <div class="buttons is-centered" style="margin-top: 0.5rem; display: inline-flex; gap: 0.5rem;">
        <button class="button is-rounded is-small is-outlined" onclick="dualStreamGoTo(0)">Step 1</button>
        <button class="button is-rounded is-small is-outlined" onclick="dualStreamGoTo(1)">Step 2</button>
        <button class="button is-rounded is-small is-outlined" onclick="dualStreamGoTo(2)">Step 3</button>
        <button class="button is-rounded is-small is-outlined" onclick="dualStreamGoTo(3)">Step 4</button>
      </div>
    </div>

    <!-- Dual Stream Columns -->
    <div class="columns">
      <!-- Semantic Stream (left) -->
      <div class="column is-half">
        <div class="box ds-stream ds-semantic" id="ds-semantic">
          <h4 class="title is-5" style="color: #2563eb;">Semantic Stream</h4>
          <p class="ds-label">Reasoning &amp; Intent</p>
          <div class="ds-content" id="ds-semantic-content">
            <p class="has-text-grey-light">Press "Play All" or click a step to begin...</p>
          </div>
        </div>
      </div>
      <!-- Runtime Stream (right) -->
      <div class="column is-half">
        <div class="box ds-stream ds-runtime" id="ds-runtime">
          <h4 class="title is-5" style="color: #059669;">Runtime Stream</h4>
          <p class="ds-label">Persistent State &amp; Data</p>
          <div class="ds-content" id="ds-runtime-content">
            <p class="has-text-grey-light">Waiting for initialization...</p>
          </div>
        </div>
      </div>
    </div>

    <!-- Arrow / status bar -->
    <div class="has-text-centered">
      <div class="ds-status" id="ds-status">
        <span class="tag is-medium is-light">Ready</span>
      </div>
    </div>
  </div>
</section>
```

**Step 2: Add CSS for dual-stream animation in caveagent.css**

```css
/* ===== Dual-Stream Animation ===== */
.ds-stream {
  min-height: 280px;
  transition: all 0.4s ease;
  position: relative;
  overflow: hidden;
}
.ds-semantic {
  background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%);
  border-left: 4px solid #2563eb;
}
.ds-runtime {
  background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
  border-left: 4px solid #059669;
}
.ds-stream.ds-active {
  box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.3);
  transform: scale(1.02);
}
.ds-runtime.ds-active {
  box-shadow: 0 0 0 3px rgba(5, 150, 105, 0.3);
}
.ds-label {
  font-size: 0.85rem;
  color: #6b7280;
  margin-bottom: 1rem;
}
.ds-content {
  transition: opacity 0.3s ease;
}
.ds-content .ds-step {
  animation: dsFadeIn 0.5s ease forwards;
  opacity: 0;
}
@keyframes dsFadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
.ds-content pre {
  background: rgba(0,0,0,0.05);
  padding: 0.75rem;
  border-radius: 6px;
  font-size: 0.85rem;
  overflow-x: auto;
}
.ds-content .ds-user-query {
  background: #fff;
  border: 1px solid #e5e7eb;
  border-radius: 8px;
  padding: 0.75rem;
  margin-bottom: 0.5rem;
}
.ds-content .ds-var {
  display: inline-block;
  background: #fff;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  padding: 0.4rem 0.75rem;
  margin: 0.25rem;
  font-family: monospace;
  font-size: 0.85rem;
  transition: all 0.3s ease;
}
.ds-content .ds-var.ds-updated {
  background: #fef3c7;
  border-color: #f59e0b;
  transform: scale(1.05);
}
.ds-status {
  margin-top: 1rem;
}

/* Step button active state */
.ds-step-active {
  background-color: #2563eb !important;
  color: white !important;
  border-color: #2563eb !important;
}
```

**Step 3: Add JavaScript for dual-stream interaction in caveagent.js**

```javascript
/* ===== Dual-Stream Animation ===== */
const dualStreamSteps = [
  {
    name: "Runtime Initialization",
    semantic: `<div class="ds-step"><p><strong>Step 1:</strong> Initialize runtime with variables and functions</p><pre><code>runtime = PythonRuntime(
  variables=[
    Variable("secret", "!dlrow ,olleH"),
    Variable("greeting", ""),
  ],
  functions=[Function(reverse)],
)
agent = CaveAgent(model, runtime)</code></pre></div>`,
    runtime: `<div class="ds-step"><p><strong>Runtime State S<sub>0</sub></strong></p><div class="ds-var">secret = "!dlrow ,olleH"</div><div class="ds-var">greeting = ""</div><div class="ds-var">reverse(s) -> str</div></div>`,
    status: "Inject",
    activeSide: "runtime"
  },
  {
    name: "User Query",
    semantic: `<div class="ds-step"><div class="ds-user-query"><strong>User:</strong> "Reverse the secret and store it in greeting"</div><p>LLM receives query + runtime state description (variable names, types, functions)</p></div>`,
    runtime: `<div class="ds-step"><p><strong>Runtime State S<sub>0</sub></strong> (unchanged)</p><div class="ds-var">secret = "!dlrow ,olleH"</div><div class="ds-var">greeting = ""</div><div class="ds-var">reverse(s) -> str</div></div>`,
    status: "Input -> Semantic Stream",
    activeSide: "semantic"
  },
  {
    name: "Code Generation & Execution",
    semantic: `<div class="ds-step"><p><strong>LLM generates code:</strong></p><pre><code class="language-python">greeting = reverse(secret)
print(greeting)</code></pre></div>`,
    runtime: `<div class="ds-step"><p><strong>Runtime State S<sub>1</sub></strong> (updated!)</p><div class="ds-var">secret = "!dlrow ,olleH"</div><div class="ds-var ds-updated">greeting = "Hello, world!"</div><div class="ds-var">reverse(s) -> str</div><p style="margin-top:0.5rem;color:#059669;"><strong>stdout:</strong> Hello, world!</p></div>`,
    status: "Code -> Execute -> Update State",
    activeSide: "both"
  },
  {
    name: "Observation & Response",
    semantic: `<div class="ds-step"><p><strong>Execution output fed back:</strong></p><pre>Output: Hello, world!</pre><p>LLM sees result, generates final response:</p><p style="background:#fff;padding:0.5rem;border-radius:6px;border:1px solid #e5e7eb;"><em>"I've reversed the secret. The greeting is now 'Hello, world!'"</em></p></div>`,
    runtime: `<div class="ds-step"><p><strong>Final Runtime State</strong></p><div class="ds-var">secret = "!dlrow ,olleH"</div><div class="ds-var ds-updated">greeting = "Hello, world!"</div><div class="ds-var">reverse(s) -> str</div><p style="margin-top:0.75rem;"><strong>Retrieve:</strong></p><pre>runtime.retrieve("greeting")
# -> "Hello, world!"</pre></div>`,
    status: "Complete - Objects retrievable!",
    activeSide: "both"
  }
];

let dsCurrentStep = -1;
let dsPlaying = false;
let dsTimer = null;

function dualStreamGoTo(step) {
  dsCurrentStep = step;
  const data = dualStreamSteps[step];
  document.getElementById('ds-semantic-content').innerHTML = data.semantic;
  document.getElementById('ds-runtime-content').innerHTML = data.runtime;
  document.getElementById('ds-status').innerHTML = '<span class="tag is-medium is-info is-light">' + data.status + '</span>';

  // Highlight active side
  document.getElementById('ds-semantic').classList.toggle('ds-active', data.activeSide === 'semantic' || data.activeSide === 'both');
  document.getElementById('ds-runtime').classList.toggle('ds-active', data.activeSide === 'runtime' || data.activeSide === 'both');

  // Update step button states
  document.querySelectorAll('[onclick^="dualStreamGoTo"]').forEach(function(btn, i) {
    btn.classList.toggle('ds-step-active', i === step);
  });

  // Re-highlight code blocks if Prism is loaded
  if (typeof Prism !== 'undefined') { Prism.highlightAll(); }
}

function dualStreamPlay() {
  if (dsPlaying) {
    clearInterval(dsTimer);
    dsPlaying = false;
    document.getElementById('ds-play-btn').innerHTML = '<span class="icon"><i class="fas fa-play"></i></span><span>Play All</span>';
    return;
  }
  dsPlaying = true;
  dsCurrentStep = -1;
  document.getElementById('ds-play-btn').innerHTML = '<span class="icon"><i class="fas fa-pause"></i></span><span>Pause</span>';
  dsTimer = setInterval(function() {
    dsCurrentStep++;
    if (dsCurrentStep >= dualStreamSteps.length) {
      clearInterval(dsTimer);
      dsPlaying = false;
      document.getElementById('ds-play-btn').innerHTML = '<span class="icon"><i class="fas fa-play"></i></span><span>Play All</span>';
      return;
    }
    dualStreamGoTo(dsCurrentStep);
  }, 2500);
}
```

**Step 4: Open browser, verify animation plays through all 4 steps**

Expected: Clicking "Play All" auto-advances every 2.5s. Clicking step buttons jumps directly. Left/right columns highlight appropriately. Code blocks have syntax highlighting.

**Step 5: Commit**

```bash
git add Website/index.html Website/static/css/caveagent.css Website/static/js/caveagent.js
git commit -m "feat(website): add interactive dual-stream animation section"
```

---

### Task 4: Add Section 4 — Smart Home Walkthrough

**Files:**
- Modify: `Website/index.html` (insert after Section 3)
- Modify: `Website/static/css/caveagent.css` (append)
- Modify: `Website/static/js/caveagent.js` (append)

**Step 1: Add HTML for smart home walkthrough in index.html**

Insert after Section 3's closing `</section>`:

```html
<!-- ==================== SECTION 4: SMART HOME WALKTHROUGH ==================== -->
<section class="section hero is-light">
  <div class="container is-max-desktop">
    <h2 class="title is-3 has-text-centered">Stateful Runtime in Action: Smart Home</h2>
    <p class="subtitle has-text-centered">Click each turn to see how CaveAgent maintains persistent state across interactions.</p>

    <!-- Turn Tabs -->
    <div class="tabs is-centered is-boxed is-medium" id="sh-tabs">
      <ul>
        <li class="is-active" onclick="shGoTo(0)"><a>Initial State</a></li>
        <li onclick="shGoTo(1)"><a>Turn 1: Morning Wake</a></li>
        <li onclick="shGoTo(2)"><a>Turn 2: Leave Home</a></li>
        <li onclick="shGoTo(3)"><a>Turn 3: Return Home</a></li>
      </ul>
    </div>

    <!-- Semantic area: user query + code -->
    <div class="box" id="sh-semantic" style="margin-bottom: 1.5rem;">
      <div id="sh-semantic-content"></div>
    </div>

    <!-- Runtime area: device cards -->
    <div class="columns is-multiline" id="sh-devices">
      <div class="column is-3">
        <div class="box has-text-centered sh-device" id="sh-thermostat">
          <p class="sh-device-icon" id="sh-thermostat-icon">&#x1F321;&#xFE0F;</p>
          <p class="sh-device-name">Thermostat</p>
          <p class="sh-device-value" id="sh-thermostat-value">18&deg;C</p>
        </div>
      </div>
      <div class="column is-3">
        <div class="box has-text-centered sh-device" id="sh-door">
          <p class="sh-device-icon" id="sh-door-icon">&#x1F512;</p>
          <p class="sh-device-name">Door</p>
          <p class="sh-device-value" id="sh-door-value">Locked</p>
        </div>
      </div>
      <div class="column is-3">
        <div class="box has-text-centered sh-device" id="sh-lights">
          <p class="sh-device-icon" id="sh-lights-icon">&#x1F4A1;</p>
          <p class="sh-device-name">Lights</p>
          <p class="sh-device-value" id="sh-lights-value">Off</p>
        </div>
      </div>
      <div class="column is-3">
        <div class="box has-text-centered sh-device" id="sh-camera">
          <p class="sh-device-icon" id="sh-camera-icon">&#x1F4F7;</p>
          <p class="sh-device-name">Camera</p>
          <p class="sh-device-value" id="sh-camera-value">Off</p>
        </div>
      </div>
    </div>
  </div>
</section>
```

**Step 2: Append CSS for smart home walkthrough to caveagent.css**

```css
/* ===== Smart Home Walkthrough ===== */
.sh-device {
  transition: all 0.4s ease;
  min-height: 160px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}
.sh-device-icon {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
  transition: all 0.3s ease;
}
.sh-device-name {
  font-weight: 600;
  font-size: 0.9rem;
  color: #374151;
}
.sh-device-value {
  font-family: monospace;
  font-size: 1.1rem;
  font-weight: 700;
  margin-top: 0.25rem;
  transition: all 0.3s ease;
}
.sh-device.sh-on {
  background: linear-gradient(135deg, #ecfdf5, #d1fae5);
  border: 2px solid #059669;
}
.sh-device.sh-off {
  background: #f9fafb;
  border: 2px solid #e5e7eb;
}
.sh-device.sh-changed {
  animation: shPulse 0.6s ease;
}
@keyframes shPulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.08); }
  100% { transform: scale(1); }
}
#sh-tabs ul li {
  cursor: pointer;
}
#sh-tabs ul li.is-active a {
  color: #2563eb;
  border-bottom-color: #2563eb;
}
```

**Step 3: Append JavaScript for smart home walkthrough to caveagent.js**

```javascript
/* ===== Smart Home Walkthrough ===== */
const shTurns = [
  {
    semantic: '<p style="color:#888;"><em>Runtime initialized with smart home devices.</em></p><pre><code class="language-python">runtime = PythonRuntime(\n  variables=[\n    Variable("thermostat", Thermostat(18)),\n    Variable("door_lock", Lock(locked=True)),\n    Variable("lights", Lights(on=False)),\n    Variable("camera", Camera(recording=False)),\n  ]\n)\nagent = CaveAgent(model, runtime)</code></pre>',
    devices: {
      thermostat: { value: "18\u00b0C", on: false },
      door: { value: "Locked", on: false, icon: "\uD83D\uDD12" },
      lights: { value: "Off", on: false },
      camera: { value: "Off", on: false }
    }
  },
  {
    semantic: '<div class="ds-user-query"><strong>User:</strong> "Good morning, start wake routine."</div><p><strong>LLM generates:</strong></p><pre><code class="language-python">if thermostat.temp < 20:\n    thermostat.set(22)\nlights.turn_on()\ndoor_lock.unlock()</code></pre>',
    devices: {
      thermostat: { value: "22\u00b0C", on: true },
      door: { value: "Unlocked", on: true, icon: "\uD83D\uDD13" },
      lights: { value: "On", on: true },
      camera: { value: "Off", on: false }
    }
  },
  {
    semantic: '<div class="ds-user-query"><strong>User:</strong> "I\'m leaving for work."</div><p><strong>LLM generates:</strong></p><pre><code class="language-python">lights.turn_off()\nthermostat.set(16)\ncamera.start_recording()\nif not door_lock.is_locked:\n    door_lock.lock()</code></pre>',
    devices: {
      thermostat: { value: "16\u00b0C", on: false },
      door: { value: "Locked", on: false, icon: "\uD83D\uDD12" },
      lights: { value: "Off", on: false },
      camera: { value: "Recording", on: true }
    }
  },
  {
    semantic: '<div class="ds-user-query"><strong>User:</strong> "I\'m back home."</div><p><strong>LLM generates:</strong></p><pre><code class="language-python">if camera.is_recording:\n    camera.stop()\nif door_lock.is_locked:\n    door_lock.unlock()\nthermostat.set(22)\nlights.turn_on()</code></pre>',
    devices: {
      thermostat: { value: "22\u00b0C", on: true },
      door: { value: "Unlocked", on: true, icon: "\uD83D\uDD13" },
      lights: { value: "On", on: true },
      camera: { value: "Off", on: false }
    }
  }
];

let shCurrent = 0;

function shGoTo(turn) {
  shCurrent = turn;
  var data = shTurns[turn];

  // Update tabs
  var tabs = document.querySelectorAll('#sh-tabs ul li');
  tabs.forEach(function(tab, i) { tab.classList.toggle('is-active', i === turn); });

  // Update semantic content
  document.getElementById('sh-semantic-content').innerHTML = data.semantic;

  // Update devices
  var deviceNames = ['thermostat', 'door', 'lights', 'camera'];
  deviceNames.forEach(function(name) {
    var el = document.getElementById('sh-' + name);
    var valEl = document.getElementById('sh-' + name + '-value');
    var d = data.devices[name];
    var oldVal = valEl.textContent;

    valEl.innerHTML = d.value;
    el.className = 'box has-text-centered sh-device ' + (d.on ? 'sh-on' : 'sh-off');

    if (d.icon) {
      document.getElementById('sh-' + name + '-icon').textContent = d.icon;
    }

    // Pulse animation on change
    if (oldVal !== d.value) {
      el.classList.add('sh-changed');
      setTimeout(function() { el.classList.remove('sh-changed'); }, 600);
    }
  });

  if (typeof Prism !== 'undefined') { Prism.highlightAll(); }
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() { shGoTo(0); });
```

**Step 4: Verify in browser**

Expected: Clicking tabs switches between turns. Device cards animate (pulse + color change). Code blocks have Python syntax highlighting. All 4 turns work correctly with correct state values.

**Step 5: Commit**

```bash
git add Website/index.html Website/static/css/caveagent.css Website/static/js/caveagent.js
git commit -m "feat(website): add interactive smart home walkthrough section"
```

---

### Task 5: Add Section 5 — Image Carousel

**Files:**
- Modify: `Website/index.html` (insert after Section 4)

**Step 1: Add carousel HTML in index.html**

Insert after Section 4's closing `</section>`:

```html
<!-- ==================== SECTION 5: IMAGE CAROUSEL ==================== -->
<section class="hero is-small">
  <div class="hero-body">
    <div class="container">
      <h2 class="title is-3 has-text-centered">Applications &amp; Case Studies</h2>
      <div id="results-carousel" class="carousel results-carousel">
        <div class="item">
          <img src="static/images/downstream_applications.png" alt="Key advantages of CaveAgent" loading="lazy"/>
          <h2 class="subtitle has-text-centered">Key Advantages: UI Rendering, Validation, RL Rewards, Multi-Agent Handoff, Benchmarking</h2>
        </div>
        <div class="item">
          <img src="static/images/town_simulation.png" alt="Multi-agent town simulation" loading="lazy"/>
          <h2 class="subtitle has-text-centered">Town Simulation: Stateful Runtime-Mediated Multi-Agent Collaboration</h2>
        </div>
        <div class="item">
          <img src="static/images/skills.png" alt="Agent Skills architecture" loading="lazy"/>
          <h2 class="subtitle has-text-centered">Agent Skills: Runtime-integrated skill management extending the open standard</h2>
        </div>
        <div class="item">
          <img src="static/images/automl.png" alt="AutoML multi-agent case study" loading="lazy"/>
          <h2 class="subtitle has-text-centered">AutoML: Multi-Agent Coordination for automated machine learning pipelines</h2>
        </div>
        <div class="item">
          <img src="static/images/geospatial.png" alt="Geospatial analysis case study" loading="lazy"/>
          <h2 class="subtitle has-text-centered">Geospatial Analysis: Complex data processing with persistent runtime state</h2>
        </div>
      </div>
    </div>
  </div>
</section>
```

**Step 2: Verify carousel renders and auto-advances**

Expected: Bulma carousel initializes (via template's index.js), images rotate, captions display.

**Step 3: Commit**

```bash
git add Website/index.html
git commit -m "feat(website): add image carousel with case studies"
```

---

### Task 6: Add Section 6 — Experimental Results with Tabs

**Files:**
- Modify: `Website/index.html` (insert after Section 5)
- Modify: `Website/static/css/caveagent.css` (append)
- Modify: `Website/static/js/caveagent.js` (append)

**Step 1: Add HTML for tabbed results in index.html**

Insert after Section 5's closing `</section>`:

```html
<!-- ==================== SECTION 6: EXPERIMENTAL RESULTS ==================== -->
<section class="section hero is-light">
  <div class="container is-max-desktop">
    <h2 class="title is-3 has-text-centered">Experimental Results</h2>

    <!-- Tabs -->
    <div class="tabs is-centered is-boxed" id="results-tabs">
      <ul>
        <li class="is-active" onclick="resultsTab(0)"><a>Tau&sup2;-Bench</a></li>
        <li onclick="resultsTab(1)"><a>Token Efficiency</a></li>
        <li onclick="resultsTab(2)"><a>Type Proficiency</a></li>
        <li onclick="resultsTab(3)"><a>Multi-Turn Analysis</a></li>
      </ul>
    </div>

    <!-- Tab content panels -->
    <div class="results-panel" id="results-panel-0">
      <p class="has-text-centered" style="margin-bottom: 1rem;">
        Performance on <strong>Tau<sup>2</sup>-bench (Retail domain)</strong> across 6 SOTA LLMs. CaveAgent achieves <strong>+10.5% average success rate improvement</strong>.
      </p>
      <div class="table-container">
        <table class="table is-striped is-hoverable is-fullwidth">
          <thead>
            <tr>
              <th>Model</th>
              <th>Baseline SR</th>
              <th>CaveAgent SR</th>
              <th>Improvement</th>
            </tr>
          </thead>
          <tbody>
            <tr><td>GPT-4o</td><td>31.0%</td><td>37.0%</td><td class="has-text-success has-text-weight-bold">+6.0%</td></tr>
            <tr><td>Claude 3.5 Sonnet</td><td>28.5%</td><td>40.5%</td><td class="has-text-success has-text-weight-bold">+12.0%</td></tr>
            <tr><td>Gemini 2.0 Flash</td><td>21.5%</td><td>31.5%</td><td class="has-text-success has-text-weight-bold">+10.0%</td></tr>
            <tr><td>Gemini 2.5 Pro</td><td>38.5%</td><td>46.5%</td><td class="has-text-success has-text-weight-bold">+8.0%</td></tr>
            <tr><td>DeepSeek-V3</td><td>20.5%</td><td>33.5%</td><td class="has-text-success has-text-weight-bold">+13.0%</td></tr>
            <tr><td>Qwen 2.5 72B</td><td>19.0%</td><td>33.0%</td><td class="has-text-success has-text-weight-bold">+14.0%</td></tr>
          </tbody>
        </table>
      </div>
      <p class="has-text-centered has-text-grey"><em>SR = Success Rate. Full results including Airline domain available in the paper.</em></p>
    </div>

    <div class="results-panel" id="results-panel-1" style="display:none;">
      <p class="has-text-centered" style="margin-bottom: 1rem;">
        CaveAgent achieves <strong>28.4% reduction</strong> in total token consumption and <strong>59% reduction</strong> on data-intensive tasks by delegating state to the persistent runtime.
      </p>
      <div class="has-text-centered">
        <img src="static/images/token_consumption.png" alt="Token consumption comparison" style="max-width: 100%; border: 1px solid #ddd; border-radius: 4px;" loading="lazy">
      </div>
    </div>

    <div class="results-panel" id="results-panel-2" style="display:none;">
      <p class="has-text-centered" style="margin-bottom: 1rem;">
        Type proficiency radar chart across multiple data types showing CaveAgent's consistent advantage.
      </p>
      <div class="has-text-centered">
        <img src="static/images/type_proficiency_multi_radar.png" alt="Type proficiency radar chart" style="max-width: 100%; border: 1px solid #ddd; border-radius: 4px;" loading="lazy">
      </div>
    </div>

    <div class="results-panel" id="results-panel-3" style="display:none;">
      <p class="has-text-centered" style="margin-bottom: 1rem;">
        Multi-turn interaction analysis showing improved state consistency across conversation turns.
      </p>
      <div class="has-text-centered">
        <img src="static/images/multi_turn_bar.png" alt="Multi-turn analysis bar chart" style="max-width: 100%; border: 1px solid #ddd; border-radius: 4px;" loading="lazy">
      </div>
    </div>
  </div>
</section>
```

**Step 2: Append CSS for result tabs to caveagent.css**

```css
/* ===== Result Tabs ===== */
#results-tabs ul li {
  cursor: pointer;
}
#results-tabs ul li.is-active a {
  color: #2563eb;
  border-bottom-color: #2563eb;
}
.results-panel {
  animation: dsFadeIn 0.3s ease;
}
```

**Step 3: Append JavaScript for tab switching to caveagent.js**

```javascript
/* ===== Results Tabs ===== */
function resultsTab(index) {
  // Toggle tab active state
  var tabs = document.querySelectorAll('#results-tabs ul li');
  tabs.forEach(function(tab, i) { tab.classList.toggle('is-active', i === index); });

  // Toggle panel visibility
  for (var i = 0; i < 4; i++) {
    var panel = document.getElementById('results-panel-' + i);
    panel.style.display = (i === index) ? 'block' : 'none';
  }
}
```

**Step 4: Verify all 4 tabs switch correctly**

Expected: Clicking each tab shows the corresponding panel. Table renders with colored improvement values. Image panels display (or show placeholder if PDFs not yet converted).

**Step 5: Commit**

```bash
git add Website/index.html Website/static/css/caveagent.css Website/static/js/caveagent.js
git commit -m "feat(website): add tabbed experimental results section"
```

---

### Task 7: Add Sections 7-8 — Code Quick Start + BibTeX + Footer

**Files:**
- Modify: `Website/index.html` (insert after Section 6, before `</body>`)

**Step 1: Add HTML for Quick Start, BibTeX, and Footer**

Insert after Section 6's closing `</section>`:

```html
<!-- ==================== SECTION 7: GET STARTED ==================== -->
<section class="section">
  <div class="container is-max-desktop">
    <h2 class="title is-3 has-text-centered">Get Started</h2>
    <div class="columns is-centered">
      <div class="column is-four-fifths">
        <div class="box">
          <h4 class="title is-5">Installation</h4>
          <div style="position: relative;">
            <pre><code class="language-python">pip install 'cave-agent[all]'</code></pre>
          </div>

          <h4 class="title is-5" style="margin-top: 1.5rem;">Hello World</h4>
          <pre><code class="language-python">import asyncio
from cave_agent import CaveAgent
from cave_agent.runtime import PythonRuntime, Variable, Function
from cave_agent.models import LiteLLMModel

model = LiteLLMModel(
    model_id="model-id",
    api_key="your-api-key",
    custom_llm_provider="openai"
)

async def main():
    def reverse(s: str) -> str:
        """Reverse a string"""
        return s[::-1]

    runtime = PythonRuntime(
        variables=[
            Variable("secret", "!dlrow ,olleH", "A reversed message"),
            Variable("greeting", "", "Store the reversed message"),
        ],
        functions=[Function(reverse)],
    )
    agent = CaveAgent(model, runtime=runtime)
    response = await agent.run("Reverse the secret")
    print(runtime.retrieve("greeting"))  # Hello, world!

asyncio.run(main())</code></pre>
        </div>

        <div class="has-text-centered" style="margin-top: 1.5rem;">
          <a href="https://github.com/acodercat/cave-agent" target="_blank" class="button is-rounded is-dark">
            <span class="icon"><i class="fab fa-github"></i></span>
            <span>Full Documentation</span>
          </a>
          <a href="https://pypi.org/project/cave-agent" target="_blank" class="button is-rounded is-outlined">
            <span class="icon"><i class="fas fa-cube"></i></span>
            <span>PyPI Package</span>
          </a>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- ==================== SECTION 8: BIBTEX + FOOTER ==================== -->
<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <div class="bibtex-header">
      <h2 class="title">BibTeX</h2>
      <button class="copy-bibtex-btn" onclick="copyBibTeX()" title="Copy BibTeX to clipboard">
        <i class="fas fa-copy"></i>
        <span class="copy-text">Copy</span>
      </button>
    </div>
    <pre id="bibtex-code"><code>@article{ran2026caveagent,
  title={CaveAgent: Transforming LLMs into Stateful Runtime Operators},
  author={Ran, Maohao and Wan, Zhenglin and Lin, Cooper and Zhang, Yanting and others},
  journal={arXiv preprint arXiv:2601.01569},
  year={2026}
}</code></pre>
  </div>
</section>

<footer class="footer">
  <div class="container">
    <div class="columns is-centered">
      <div class="column is-8">
        <div class="content has-text-centered">
          <p>
            This page was built using the <a href="https://github.com/eliahuhorwitz/Academic-project-page-template" target="_blank">Academic Project Page Template</a>.
            Licensed under <a href="https://creativecommons.org/licenses/by-sa/4.0/" target="_blank">CC BY-SA 4.0</a>.
          </p>
        </div>
      </div>
    </div>
  </div>
</footer>
```

**Step 2: Verify all sections render end-to-end**

```bash
cd /mnt/data/wanzl/cave-agent/Website
python3 -m http.server 8080
# Visit http://localhost:8080
```

Expected: Full page scrolls through all 8 sections. Code blocks highlighted. BibTeX copy works. Footer renders.

**Step 3: Commit**

```bash
git add Website/index.html
git commit -m "feat(website): add Get Started, BibTeX, and Footer sections"
```

---

### Task 8: Convert PDF figures to PNG

**Files:**
- Create: `Website/static/images/token_consumption.png`
- Create: `Website/static/images/type_proficiency_multi_radar.png`
- Create: `Website/static/images/multi_turn_bar.png`
- Create: `Website/static/images/steps_success.png`
- Create: `Website/static/images/benchmark_combined.png`
- Create: `Website/static/images/vars_count_heatmap.png`

**Step 1: Convert all PDF figures**

```bash
cd /mnt/data/wanzl/cave-agent

# Try pdftoppm first (from poppler-utils)
for pdf in token_consumption steps_success multi_turn_bar type_proficiency_multi_radar benchmark_combined vars_count_heatmap; do
  if [ -f "Website/Cave_agent_overleaf_src/figures/CaveAgent/${pdf}.pdf" ]; then
    pdftoppm -png -r 300 -singlefile \
      "Website/Cave_agent_overleaf_src/figures/CaveAgent/${pdf}.pdf" \
      "Website/static/images/${pdf}"
    echo "Converted: ${pdf}"
  fi
done

# Fallback: If pdftoppm unavailable, try ImageMagick convert
# convert -density 300 input.pdf -quality 95 output.png
```

**Step 2: Verify PNGs exist and are reasonable size**

```bash
ls -lh Website/static/images/*.png
```

Expected: All 6 PNG files present, each 50KB-2MB, 300 DPI quality.

**Step 3: Commit**

```bash
git add Website/static/images/*.png
git commit -m "feat(website): convert PDF figures to PNG for web display"
```

---

### Task 9: Final polish — responsive check, accessibility, meta tags

**Files:**
- Modify: `Website/index.html` (minor tweaks)
- Modify: `Website/static/css/caveagent.css` (responsive fixes)

**Step 1: Add responsive CSS fixes to caveagent.css**

```css
/* ===== Responsive ===== */
@media screen and (max-width: 768px) {
  .ds-stream {
    min-height: 200px;
  }
  .sh-device {
    min-height: 120px;
  }
  .sh-device-icon {
    font-size: 1.8rem;
  }
  .institution-logos img {
    height: 25px !important;
  }
  #sh-tabs .tabs ul {
    flex-wrap: wrap;
  }
  #sh-tabs .tabs ul li {
    font-size: 0.8rem;
  }
}
```

**Step 2: Test in browser at different viewport sizes**

Check: 320px (mobile), 768px (tablet), 1200px (desktop). All sections readable, no overflow, carousels swipeable.

**Step 3: Verify all links work**

Check: arXiv, GitHub, PyPI buttons open correct URLs. BibTeX copy button works. Footer link works.

**Step 4: Commit**

```bash
git add Website/
git commit -m "feat(website): responsive fixes and final polish"
```

---

### Task 10: Verify full site end-to-end

**Step 1: Start local server and walk through all sections**

```bash
cd /mnt/data/wanzl/cave-agent/Website
python3 -m http.server 8080
```

Checklist:
- [ ] Hero: Logo, title, authors, institution logos, 4 link buttons
- [ ] Teaser: Framework image displays with caption
- [ ] Abstract: Text renders, metrics bolded
- [ ] Dual-Stream: "Play All" animates 4 steps, manual step buttons work
- [ ] Smart Home: 4 tabs switch, device cards animate, code highlighted
- [ ] Carousel: 5 images rotate, captions show
- [ ] Results: 4 tabs switch, table renders, images display
- [ ] Get Started: Code highlighted, copy works
- [ ] BibTeX: Copy button works
- [ ] Footer: Attribution link works
- [ ] Mobile: All sections responsive at 320px width

**Step 2: Final commit if any fixes needed**

```bash
git add Website/
git commit -m "feat(website): complete CaveAgent paper website"
```
