/* ===== Dual-Stream Animation ===== */
const dualStreamSteps = [
  {
    name: "Runtime Initialization",
    semantic: '<div class="ds-step"><p><strong>Step 1:</strong> Initialize runtime with variables and functions</p><pre><code>runtime = PythonRuntime(\n  variables=[\n    Variable("secret", "!dlrow ,olleH"),\n    Variable("greeting", ""),\n  ],\n  functions=[Function(reverse)],\n)\nagent = CaveAgent(model, runtime)</code></pre></div>',
    runtime: '<div class="ds-step"><p><strong>Runtime State S<sub>0</sub></strong></p><div class="ds-var">secret = "!dlrow ,olleH"</div><div class="ds-var">greeting = ""</div><div class="ds-var">reverse(s) -> str</div></div>',
    status: "Inject",
    activeSide: "runtime"
  },
  {
    name: "User Query",
    semantic: '<div class="ds-step"><div class="ds-user-query"><strong>User:</strong> "Reverse the secret and store it in greeting"</div><p>LLM receives query + runtime state description (variable names, types, functions)</p></div>',
    runtime: '<div class="ds-step"><p><strong>Runtime State S<sub>0</sub></strong> (unchanged)</p><div class="ds-var">secret = "!dlrow ,olleH"</div><div class="ds-var">greeting = ""</div><div class="ds-var">reverse(s) -> str</div></div>',
    status: "Input -> Semantic Stream",
    activeSide: "semantic"
  },
  {
    name: "Code Generation & Execution",
    semantic: '<div class="ds-step"><p><strong>LLM generates code:</strong></p><pre><code class="language-python">greeting = reverse(secret)\nprint(greeting)</code></pre></div>',
    runtime: '<div class="ds-step"><p><strong>Runtime State S<sub>1</sub></strong> (updated!)</p><div class="ds-var">secret = "!dlrow ,olleH"</div><div class="ds-var ds-updated">greeting = "Hello, world!"</div><div class="ds-var">reverse(s) -> str</div><p style="margin-top:0.5rem;color:#059669;"><strong>stdout:</strong> Hello, world!</p></div>',
    status: "Code -> Execute -> Update State",
    activeSide: "both"
  },
  {
    name: "Observation & Response",
    semantic: '<div class="ds-step"><p><strong>Execution output fed back:</strong></p><pre>Output: Hello, world!</pre><p>LLM sees result, generates final response:</p><p style="background:#fff;padding:0.5rem;border-radius:6px;border:1px solid #e5e7eb;"><em>"I\'ve reversed the secret. The greeting is now \'Hello, world!\'"</em></p></div>',
    runtime: '<div class="ds-step"><p><strong>Final Runtime State</strong></p><div class="ds-var">secret = "!dlrow ,olleH"</div><div class="ds-var ds-updated">greeting = "Hello, world!"</div><div class="ds-var">reverse(s) -> str</div><p style="margin-top:0.75rem;"><strong>Retrieve:</strong></p><pre>runtime.retrieve("greeting")\n# -> "Hello, world!"</pre></div>',
    status: "Complete - Objects retrievable!",
    activeSide: "both"
  }
];

var dsCurrentStep = -1;
var dsPlaying = false;
var dsTimer = null;

function dualStreamGoTo(step) {
  dsCurrentStep = step;
  var data = dualStreamSteps[step];
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

var shCurrent = 0;

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

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() { shGoTo(0); });
