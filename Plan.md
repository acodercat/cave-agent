# CaveAgent Paper Website — Design Plan

**Date:** 2026-03-07
**Approach:** Academic Project Page Template + Enhanced Interactions (Plan B)

---

## Tech Stack

| Layer | Choice |
|-------|--------|
| Base template | [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template) |
| CSS framework | Bulma (template built-in) |
| Font | Inter (template built-in) |
| Code highlighting | Prism.js (lightweight, Python support) |
| Animations | Native CSS transitions + vanilla JS |
| Carousel | bulma-carousel (template built-in) |
| Deployment | GitHub Pages (static HTML) |
| Charts | PDF pre-converted to PNG, placed in `static/images/` |

---

## Site Structure (8 Sections)

### Section 1: Hero

- CaveAgent Logo (Logo-CaveAgent.png) centered at top
- Title: "CaveAgent: Transforming LLMs into Stateful Runtime Operators"
- Tagline: "From text-in-text-out to (text&object)-in-(text&object)-out"
- Author list with personal page links (* equal contribution, dagger corresponding)
- Institution logo row: HKUST, HKBU, NUS, HKU, CMU, NTU, Imperial, Princeton, UNC, Harvard
- Button row: `arXiv Paper` | `GitHub Code` | `PyPI Package`

### Section 2: Teaser + Abstract

- Full-width `framework.png` (dual-stream architecture) with gray border and caption
- Abstract on light gray background, key metrics bold-highlighted:
  - +10.5% success rate on retail tasks
  - 28.4% reduction in total token consumption
  - 59% token reduction on data-intensive tasks

### Section 3: Interactive Dual-Stream Animation (Feature Interaction 1)

- Title: "How CaveAgent Works"
- Step-by-step animated walkthrough based on framework.png logic:
  - **Step 1** — Runtime Initialization: code block fades in showing `PythonRuntime(variables=[...], functions=[...])`, arrow animates toward Runtime Stream
  - **Step 2** — User Query -> Semantic Stream: user input appears, LLM reasoning highlighted
  - **Step 3** — Code Generation -> Runtime Execution: code flows from LLM, Runtime state updates (variable value change animation)
  - **Step 4** — Observation -> Next Turn: execution result flows back to Semantic Stream, loop
- Two-column layout: left = Semantic Stream (light blue bg), right = Runtime Stream (light green bg), arrows connecting
- Bottom controls: "Play All" auto-play + manual step buttons (1/2/3/4)

### Section 4: Smart Home Walkthrough (Feature Interaction 2)

- Title: "Stateful Runtime in Action: Smart Home Demo"
- Based on smarthome.png content, made into interactive step walkthrough
- Top tab buttons: `Initial State` | `Turn 1: Morning Wake` | `Turn 2: Leave Home` | `Turn 3: Return Home`
- Upper area (Semantic Stream): user command + LLM-generated code with syntax highlighting
- Lower area (Runtime Stream): 4 device cards (Thermostat, Door, Lights, Camera), clicking different Turns triggers real-time state changes (values, colors, icon transitions)
- Device state changes use CSS transitions for smooth animation (number rolling, bulb on/off, lock toggle)

### Section 5: Image Carousel — More Examples & Applications

- Standard Bulma carousel component
- Content:
  1. `downstream_applications.png` — Key Advantages
  2. `town_simulation.png` — Multi-Agent Town Simulation
  3. `Skills.png` — Agent Skills Architecture
  4. `AutoML.png` — AutoML Multi-Agent Case Study
  5. `Geospacial_Analysis.png` — Geospatial Analysis Case Study
- Each image with short caption

### Section 6: Experimental Results (Tab Switching)

- Title: "Experimental Results"
- Tab bar: `Tau2-Bench Results` | `Token Efficiency` | `Type Proficiency` | `Multi-Turn Analysis`
- Tab contents:
  - **Tau2-Bench**: HTML table reproducing main_table data (CaveAgent rows highlighted blue), showing +10.5% success rate
  - **Token Efficiency**: `token_consumption.pdf` converted to PNG + key metric description (28.4% reduction)
  - **Type Proficiency**: `type_proficiency_multi_radar.pdf` converted to PNG radar chart
  - **Multi-Turn**: `multi_turn_bar.pdf` converted to PNG bar chart
- PDF figures must be pre-converted to PNG for web display

### Section 7: Code Quick Start

- Title: "Get Started"
- Install command: `pip install 'cave-agent[all]'` with copy button
- Hello World code block (from README) with Prism.js Python syntax highlighting
- Bottom link buttons: `Full Documentation` -> GitHub README | `PyPI` -> PyPI page

### Section 8: BibTeX + Footer

- BibTeX citation block + copy button (template standard feature)
- Footer: template attribution + CC license + institution logo row (small)

---

## File Structure

```
Website/
├── index.html                    # Main page
├── static/
│   ├── css/
│   │   ├── bulma.min.css        # Bulma framework
│   │   ├── bulma-carousel.min.css
│   │   ├── bulma-slider.min.css
│   │   ├── index.css            # Template base styles
│   │   └── caveagent.css        # Custom CaveAgent styles (animations, tabs, walkthrough)
│   ├── js/
│   │   ├── index.js             # Template base JS
│   │   ├── bulma-carousel.min.js
│   │   ├── bulma-slider.min.js
│   │   ├── prism.min.js         # Code highlighting
│   │   └── caveagent.js         # Custom interactions (dual-stream animation, smart home walkthrough, tabs)
│   ├── images/
│   │   ├── logo.png             # CaveAgent logo
│   │   ├── framework.png        # Dual-stream architecture (teaser)
│   │   ├── smarthome.png        # Smart home demo
│   │   ├── town_simulation.png
│   │   ├── downstream_applications.png
│   │   ├── skills.png
│   │   ├── automl.png
│   │   ├── geospatial.png
│   │   ├── token_consumption.png # Converted from PDF
│   │   ├── type_proficiency.png  # Converted from PDF
│   │   ├── multi_turn_bar.png    # Converted from PDF
│   │   ├── benchmark_combined.png # Converted from PDF
│   │   └── institution_logos/    # University logos
│   └── favicon.ico
└── .nojekyll                     # GitHub Pages config
```

---

## Key Data from Paper (for populating content)

### Authors
Maohao Ran*1,2, Zhenglin Wan*1,3, Cooper Lin4, Yanting Zhang1, Hongyu Xin1, Hongwei Fan7, Yibo Xu1, Beier Luo4, Yaxin Zhou5, Wangbo Zhao3, Lijie Yang8, Lang Feng6, Fuchao Yang6, Jingxuan Wu9, Yiqiao Huang10, Chendong Ma2, Dailing Jiang2, Jianbo Deng1, Sirui Han1, Yang You3, Bo An6, Yike Guo1, Jun Song1,2†

### Affiliations
1-HKUST, 2-HKBU, 3-NUS, 4-HKU, 5-CMU, 6-NTU Singapore, 7-Imperial College London, 8-Princeton, 9-UNC Chapel Hill, 10-Harvard

### Links
- arXiv: https://arxiv.org/abs/2601.01569
- GitHub: https://github.com/acodercat/cave-agent
- PyPI: https://pypi.org/project/cave-agent

### BibTeX
```bibtex
@article{ran2026caveagent,
  title={CaveAgent: Transforming LLMs into Stateful Runtime Operators},
  author={Ran, Maohao and Wan, Zhenglin and Lin, Cooper and Zhang, Yanting and others},
  journal={arXiv preprint arXiv:2601.01569},
  year={2026}
}
```

### Smart Home Walkthrough Data

**Initial State:**
- Thermostat: 18°C
- Door: Locked
- Lights: Off
- Camera: Off

**Turn 1 (Morning Wake) — User: "Good morning, start wake routine."**
```python
if thermostat.temp < 20:
    thermostat.set(22)
lights.turn_on()
door_lock.unlock()
```
State: Thermostat 22°C, Door Unlocked, Lights On, Camera Off

**Turn 2 (Leave Home) — User: "I'm leaving for work."**
```python
lights.turn_off()
thermostat.set(16)
camera.start_recording()
if not door_lock.is_locked:
    door_lock.lock()
```
State: Thermostat 16°C, Door Locked, Lights Off, Camera On

**Turn 3 (Return Home) — User: "I'm back home."**
```python
if camera.is_recording:
    camera.stop()
if door_lock.is_locked:
    door_lock.unlock()
thermostat.set(22)
lights.turn_on()
```
State: Thermostat 22°C, Door Unlocked, Lights On, Camera Off
