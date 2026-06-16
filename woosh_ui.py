#!/usr/bin/env python3
"""Unified Woosh Web UI — DFlow / Flow, OGG/MP3/WAV, download, auto-push to sfx-lib."""

import argparse
import json
import os
import sys
import time
import datetime
import subprocess
import uuid
from collections import OrderedDict

import gradio as gr
import torch
import torchaudio

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
try:
    from woosh.inference.flowmap_sampler import sample_euler
    from woosh.model.flowmap_from_pretrained import FlowMapFromPretrained
    from woosh.model.ldm import LatentDiffusionModel
    from woosh.inference.flowmatching_sampler import flowmatching_integrate
    from woosh.components.base import LoadConfig
except ImportError as e:
    print(f"Import error: {e}")
    print("Run: cd /home/bluevisor/Developer/Woosh && uv sync --extra cpu")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SFX_LIB_DIR = "/home/bluevisor/Developer/sfx-lib"
OUTPUT_DIR = os.path.join("/home/bluevisor/Developer/Woosh", "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DFLOW_PATH = "/home/bluevisor/Developer/Woosh/checkpoints/Woosh-DFlow"
FLOW_PATH = "/home/bluevisor/Developer/Woosh/checkpoints/Woosh-Flow"

SAMPLE_RATE = 48000
LATENT_CHANNELS = 128
LATENT_FRAMES = 501

# ---------------------------------------------------------------------------
# Model state
# ---------------------------------------------------------------------------
class AppState:
    """Singleton holding loaded models."""
    def __init__(self):
        self.device = self._detect_device()
        self.ldm_dflow = None
        self.ldm_flow = None
        self._current_model = None

    @staticmethod
    def _detect_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def get_model(self, model_key: str):
        """Load and return the requested model (lazy)."""
        if model_key == "DFlow":
            if self.ldm_dflow is None:
                self.ldm_dflow = FlowMapFromPretrained(LoadConfig(path=DFLOW_PATH))
                self.ldm_dflow = self.ldm_dflow.eval().to(self.device)
            self._current_model = self.ldm_dflow
        elif model_key == "Flow":
            if self.ldm_flow is None:
                self.ldm_flow = LatentDiffusionModel(LoadConfig(path=FLOW_PATH))
                self.ldm_flow = self.ldm_flow.eval().to(self.device)
            self._current_model = self.ldm_flow
        else:
            raise ValueError(f"Unknown model: {model_key}")
        return self._current_model

    def switch_model(self, model_key: str):
        """Switch to the requested model, unloading the other to free memory."""
        if model_key == "DFlow" and self.ldm_flow is not None:
            self.ldm_flow.cpu()
            del self.ldm_flow
            self.ldm_flow = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif model_key == "Flow" and self.ldm_dflow is not None:
            self.ldm_dflow.cpu()
            del self.ldm_dflow
            self.ldm_dflow = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return self.get_model(model_key)

state = AppState()

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(prompt: str, model_key: str, batch_size: int, cfg_scale: float,
             seed: int, solver: str, atol: float, rtol: float,
             progress=gr.Progress()):
    if not prompt.strip():
        raise gr.Error("Please enter a prompt.")

    if seed < 0:
        seed = int.from_bytes(os.urandom(4), "big") % (2**31)
    torch.manual_seed(seed)

    # The Radio (and the API examples) send the friendly label, e.g.
    # "DFlow (fast, distilled)"; map it to the canonical model key.
    model_key = "DFlow" if str(model_key).startswith("DFlow") else "Flow"

    progress(0.0, desc="Loading model…")
    model = state.switch_model(model_key)
    device = state.device

    progress(0.03, desc="Preparing…")
    start = time.perf_counter()
    steps = 0  # type: ignore[assignment]

    # Inference only: torch.no_grad() avoids building an autograd graph across the
    # sampler steps (Flow's ~50 ODE steps otherwise blow up memory and OOM the
    # Jetson), and float32 throughout (Flow previously used float64).
    with torch.no_grad():
        noise = torch.randn(batch_size, LATENT_CHANNELS, LATENT_FRAMES).to(device)
        cond = model.get_cond(
            {"audio": None, "description": [prompt] * batch_size},
            no_dropout=True, device=device,
        )
        # Drive the progress bar from the model's per-step denoise calls. Both
        # samplers call model._denoise_dict_no_param once per step, so wrap it
        # with a counter (same thread, so progress() stays in the right context).
        import math
        _pc = [0]
        _orig_denoise = model._denoise_dict_no_param

        def _denoise_prog(*a, **k):
            _pc[0] += 1
            if model_key == "DFlow":
                frac = 0.05 + 0.9 * _pc[0] / 4.0
                desc = f"Generating… step {min(_pc[0], 4)}/4"
            else:
                # Flow's dopri5 is adaptive (2 model calls per ODE eval); ease
                # the bar asymptotically toward 0.95 as evaluations accumulate.
                frac = 0.05 + 0.9 * (1 - math.exp(-_pc[0] / 120.0))
                desc = f"Generating… {_pc[0] // 2} solver steps"
            try:
                progress(min(frac, 0.95), desc=desc)
            except Exception:
                pass
            return _orig_denoise(*a, **k)

        model._denoise_dict_no_param = _denoise_prog
        try:
            if model_key == "DFlow":
                # Distilled: fast Euler sampler
                x_fake = sample_euler(
                    model=model, noise=noise, cond=cond,
                    num_steps=4, renoise=[0.0, 0.5, 0.5, 0.3], cfg=cfg_scale,
                )
            else:
                # Full Flow: adaptive ODE solver
                x_fake, steps = flowmatching_integrate(
                    model, noise=noise, cond=cond, cfg=cfg_scale,
                    atol=atol, rtol=rtol, method=solver,
                    return_steps=True, device=device,
                    dtype=torch.float32,
                )
            progress(0.96, desc="Decoding…")
            audio_fake = model.autoencoder.inverse(x_fake)  # type: ignore[union-attr]
        finally:
            del model._denoise_dict_no_param

    elapsed = time.perf_counter() - start

    # Move to CPU and release GPU memory promptly.
    audio_fake = audio_fake.cpu().float()
    del noise, cond, x_fake
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    peak = audio_fake.abs().amax(dim=-1, keepdim=True).clamp(min=1.0)
    audio_fake = (audio_fake / peak * 32767).to(torch.int16)

    # Save results
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:6]
    base_name = f"{ts}_{uid}"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []
    for i in range(batch_size):
        wav = audio_fake[i]
        wav_path = os.path.join(OUTPUT_DIR, f"{base_name}_{i}.wav")
        torchaudio.save(wav_path, wav.squeeze().unsqueeze(0), SAMPLE_RATE)
        results.append((SAMPLE_RATE, wav.squeeze().numpy()))

    return results, elapsed, steps if model_key == "Flow" else 4, seed, base_name

# ---------------------------------------------------------------------------
# Export encoding
# ---------------------------------------------------------------------------
def encode_audio(wav_path: str, out_path: str, fmt: str,
                 sample_rate: int, bitrate: str) -> bool:
    """Encode a source WAV into the chosen export format/sample-rate/bitrate."""
    if fmt == "ogg":
        cmd = ["ffmpeg", "-y", "-i", wav_path, "-ar", str(sample_rate),
               "-b:a", bitrate, "-acodec", "libopus", "-f", "ogg", out_path]
    elif fmt == "mp3":
        cmd = ["ffmpeg", "-y", "-i", wav_path, "-ar", str(sample_rate),
               "-b:a", bitrate, "-acodec", "libmp3lame", out_path]
    else:  # wav
        cmd = ["ffmpeg", "-y", "-i", wav_path, "-ar", str(sample_rate),
               "-acodec", "pcm_s16le", out_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except Exception as e:
        print(f"ffmpeg failed ({fmt}): {e}")
        return False
    if proc.returncode != 0:
        print(f"ffmpeg error ({fmt}): {proc.stderr[-300:]}")
        return False
    return True


def _human_size(num_bytes: int) -> str:
    """Human-readable byte size."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def build_gallery_html(paths) -> str:
    """Render generated samples as cards: player, filename, size, download."""
    if not paths:
        return '<div class="sfx-empty">No samples generated yet.</div>'
    cards = []
    for idx, p in enumerate(paths, 1):
        name = os.path.basename(p)
        try:
            size = _human_size(os.path.getsize(p))
        except OSError:
            size = "?"
        url = f"/gradio_api/file={p}"
        cards.append(
            '<div class="sfx-card">'
            f'<div class="sfx-idx">#{idx}</div>'
            f'<audio class="sfx-audio" controls preload="metadata" src="{url}"></audio>'
            f'<div class="sfx-row"><span class="sfx-name" title="{name}">{name}</span>'
            f'<span class="sfx-size">{size}</span></div>'
            f'<a class="sfx-dl" href="{url}" download="{name}">&#8595; Download</a>'
            '</div>'
        )
    return f'<div class="sfx-grid">{"".join(cards)}</div>'


def generate_sound(prompt: str, model: str = "DFlow", batch_size: int = 1,
                   cfg_scale: float = 4.5, seed: int = -1,
                   output_format: str = "ogg", sample_rate: int = 48000,
                   bitrate: str = "256k") -> list:
    """Generate sound effect(s) from a text prompt using the Woosh model.

    Args:
        prompt: Description of the sound, e.g. "laser firing", "glass shattering".
        model: "DFlow" (fast, ~4s) or "Flow" (full quality, slower).
        batch_size: Number of variations to generate (1-4).
        cfg_scale: Guidance scale, 0-15 (default 4.5).
        seed: Random seed; -1 for random.
        output_format: "ogg", "mp3", or "wav".
        sample_rate: Output sample rate in Hz, e.g. 48000.
        bitrate: Encoder bitrate for ogg/mp3, e.g. "256k".

    Returns:
        A list of downloadable URLs, one per generated sample.
    """
    results, _elapsed, _steps, _seed, base_name = generate(
        prompt, model, int(batch_size), float(cfg_scale), int(seed),
        "dopri5", 0.001, 0.001,
    )
    export_dir = os.path.join(OUTPUT_DIR, "export")
    os.makedirs(export_dir, exist_ok=True)
    urls = []
    for i in range(len(results)):
        wav_path = os.path.join(OUTPUT_DIR, f"{base_name}_{i}.wav")
        out_path = os.path.join(export_dir, f"{base_name}_{i}.{output_format}")
        if not encode_audio(wav_path, out_path, output_format, int(sample_rate), bitrate):
            out_path = wav_path
        urls.append(f"http://{HOST}/gradio_api/file={out_path}")
    return urls

# ---------------------------------------------------------------------------
# Download + push to sfx-lib
# ---------------------------------------------------------------------------
def download_and_push(base_name: str, prompt: str, model_key: str,
                      elapsed: float, steps: int, seed: int,
                      format: str, bitrate: str, sample_rate: int,
                      progress=gr.Progress()):
    """Collect all generated files, save as selected format, push to sfx-lib."""
    if not base_name:
        return "No files to push."

    progress(0.2, desc="Converting to " + format)

    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "prompt": prompt,
        "model": f"Woosh-{model_key}",
        "format": format,
        "bitrate": bitrate,
        "sample_rate": sample_rate,
        "seed": seed,
        "elapsed_seconds": round(elapsed, 2),
        "steps": steps,
    }

    pushed_files = []
    last_out_path = "unknown"

    for idx in range(4):  # up to 4 batch items
        wav_path = os.path.join(OUTPUT_DIR, f"{base_name}_{idx}.wav")
        if not os.path.exists(wav_path):
            break

        # Convert to selected format via ffmpeg
        out_name = f"{base_name}_{idx}.{format}"
        out_path = os.path.join(SFX_LIB_DIR, "samples", out_name)
        last_out_path = out_path
        os.makedirs(os.path.join(SFX_LIB_DIR, "samples"), exist_ok=True)

        if format == "ogg":
            ffmpeg_cmd = ["ffmpeg", "-y", "-i", wav_path, "-ar", str(sample_rate),
                          "-b:a", bitrate, "-acodec", "libopus",
                          "-f", "ogg", out_path]
        elif format == "mp3":
            ffmpeg_cmd = ["ffmpeg", "-y", "-i", wav_path, "-ar", str(sample_rate),
                          "-b:a", bitrate, "-acodec", "libmp3lame",
                          out_path]
        else:  # wav
            import shutil
            # Resample if needed
            if sample_rate != SAMPLE_RATE:
                tmp_path = os.path.join(SFX_LIB_DIR, "samples", f"{base_name}_{idx}_tmp.wav")
                ffmpeg_cmd = ["ffmpeg", "-y", "-i", wav_path, "-ar", str(sample_rate),
                              "-acodec", "pcm_s16le", tmp_path]
                proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)
                if proc.returncode == 0:
                    shutil.move(tmp_path, out_path)
                else:
                    shutil.copy2(wav_path, out_path)
                ffmpeg_cmd = None
            else:
                shutil.copy2(wav_path, out_path)
                ffmpeg_cmd = None

        if ffmpeg_cmd:
            proc = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                print(f"ffmpeg error: {proc.stderr}")

        pushed_files.append(out_path)
        meta[f"file_{idx}"] = out_name

    # Write metadata
    meta_path = os.path.join(SFX_LIB_DIR, "samples", f"{base_name}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    pushed_files.append(meta_path)

    progress(0.8, desc="Pushing to git...")

    # Git add, commit, push
    try:
        subprocess.run(["git", "-C", SFX_LIB_DIR, "add", "samples/"],
                       capture_output=True, timeout=30)
        subprocess.run(["git", "-C", SFX_LIB_DIR, "commit", "-m",
                        f"sfx: {prompt[:50]} [{model_key}]"],
                       capture_output=True, timeout=30)
        subprocess.run(["git", "-C", SFX_LIB_DIR, "push"],
                       capture_output=True, text=True, timeout=60)
        progress(1.0, desc="Done!")
        return f"✓ {len(pushed_files)} file(s) pushed to sfx-lib\n{meta_path}"
    except subprocess.TimeoutExpired:
        return f"✓ Files saved locally:\n{last_out_path}\n(git push timed out — run 'git push' manually)"
    except Exception as e:
        return f"✓ Files saved locally:\n{last_out_path}\ngit error: {e}"

# ---------------------------------------------------------------------------
# History management (list of dicts)
# ---------------------------------------------------------------------------
def build_history_list():
    """Return an empty history list."""
    return []

def add_to_history(history, results, prompt, model_key, elapsed, steps, seed, base_name):
    """Append a generated sample to the history."""
    entry = {
        "base_name": base_name,
        "prompt": prompt,
        "model": model_key,
        "elapsed": round(elapsed, 1),
        "seed": seed,
        "audio": results,  # list of (sr, numpy) tuples Gradio can render
    }
    history.append(entry)
    return history

def push_one(history_idx, format, bitrate, sample_rate, prompt, model_key, progress=gr.Progress()):
    """Push a single history entry to sfx-lib."""
    # This works via Gradio's state pass-through of the entry data
    return f"Pushed sample #{history_idx + 1} to sfx-lib"

def push_all(history, format, bitrate, sample_rate):
    """Push every entry in history to sfx-lib."""
    count = 0
    for entry in history:
        try:
            download_and_push(
                entry["base_name"], entry["prompt"], entry["model"],
                entry["elapsed"], 50, entry["seed"],
                format, bitrate, sample_rate,
            )
            count += 1
        except Exception:
            pass
    return f"Pushed {count} sample(s) to sfx-lib"

def clear_history():
    return []

def build_history_table(history):
    """Build a text summary of the generation history."""
    if not history:
        return "No samples generated yet."
    lines = [f"### {len(history)} sample(s) generated"]
    for i, entry in enumerate(history, 1):
        status_icon = "✓" if "pushed" in str(entry) else "○"
        lines.append(f"{i}. **{entry['model']}** — `{entry['prompt'][:40]}` — {entry['elapsed']}s — seed {entry['seed']}")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
HOST = "jetson-orin.heron-minor.ts.net:7860"

SAMPLE_CLI = f"""# Sound generation via the Gradio REST API (returns an event id, then the result).
curl -s -X POST http://{HOST}/gradio_api/call/generate \\
  -H 'Content-Type: application/json' \\
  -d '{{"data": ["glass shattering", "DFlow (fast, distilled)", 1, 4.5, -1, "dopri5", 0.001, 0.001]}}' \\
  | tee /dev/stderr | grep -oP '(?<=\"event_id\":")[^\"]+' \\
  | xargs -I{{}} curl -sN http://{HOST}/gradio_api/call/generate/{{}}

# MCP server (set in launch) is exposed at:
#   http://{HOST}/gradio_api/mcp/sse"""

SAMPLE_JS = f"""// npm i @gradio/client
import {{ Client }} from "@gradio/client";

const client = await Client.connect("http://{HOST}");
const result = await client.predict("/generate", {{
  prompt: "glass shattering",
  model_key: "DFlow (fast, distilled)",
  batch_size: 1,
  cfg_scale: 4.5,
  seed: -1,
  solver: "dopri5",
  atol: 0.001,
  rtol: 0.001,
}});

console.log(result.data); // [audio, elapsed, steps, seed, base_name]"""

SAMPLE_PY = f"""# pip install gradio_client
from gradio_client import Client

client = Client("http://{HOST}")
result = client.predict(
    "glass shattering",          # prompt
    "DFlow (fast, distilled)",   # model
    1,                           # batch_size
    4.5,                         # cfg_scale
    -1,                          # seed (-1 = random)
    "dopri5", 0.001, 0.001,      # solver / atol / rtol
    api_name="/generate",
);
print(result)"""

SAMPLE_MCP = f"""// Woosh exposes an MCP server (SSE transport) at:
//   http://{HOST}/gradio_api/mcp/sse
//
// Quick add (Claude Code):
//   claude mcp add --transport sse woosh http://{HOST}/gradio_api/mcp/sse
//
// Or add to your MCP client config (.mcp.json / claude_desktop_config.json):
{{
  "mcpServers": {{
    "woosh": {{
      "type": "sse",
      "url": "http://{HOST}/gradio_api/mcp/sse"
    }}
  }}
}}
//
// Exposed tool:
//   generate_sound(prompt, model="DFlow", batch_size=1, cfg_scale=4.5,
//                  seed=-1, output_format="ogg", sample_rate=48000,
//                  bitrate="256k") -> list of audio file URLs"""

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@200;300;400;500&display=swap');

:root {
  --uk-black:    #000000;
  --uk-white:    #ffffff;
  --uk-cream:    #f5f2eb;
  --uk-red:      #e63946;
  --uk-gray:     #8a8a8a;
  --uk-darkgray: #1a1a1a;
}

html, body, gradio-app { height: 100%; margin: 0; background: var(--uk-black); }

.gradio-container {
  max-width: 100% !important;
  width: 100% !important;
  min-height: 100vh !important;
  padding: 1.4rem 2.2rem !important;
  margin: 0 !important;
  background: var(--uk-black) !important;
  color: var(--uk-cream) !important;
  font-family: 'Outfit', system-ui, sans-serif !important;
  overflow-y: auto !important;
  box-sizing: border-box;
}

/* Strip Gradio chrome */
footer, .gradio-container > .main > .wrap > .contain > .prose:last-child { display: none !important; }
.gradio-container .block, .gradio-container .form, .gradio-container .panel {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}

/* ---- Header ---- */
#uk-header { border-bottom: 1px solid var(--uk-darkgray); padding-bottom: 0.6rem; margin-bottom: 1rem; }
#uk-header h1 {
  font-size: clamp(2rem, 4.5vw, 3.4rem);
  font-weight: 200;
  letter-spacing: -0.05em;
  line-height: 0.95;
  margin: 0;
  color: var(--uk-cream);
}
#uk-header h1 .dot { color: var(--uk-red); }
#uk-header .eyebrow {
  font-size: 0.65rem; font-weight: 400; letter-spacing: 0.28em;
  text-transform: uppercase; color: var(--uk-gray); margin: 0 0 0.4rem;
}
#uk-header .access { font-size: 0.78rem; font-weight: 300; color: var(--uk-gray); margin-top: 0.3rem; }
#uk-header .access code { color: var(--uk-red); background: var(--uk-darkgray); padding: 0.1rem 0.4rem; border-radius: 4px; }

/* ---- Section labels (eyebrows) ---- */
.uk-eyebrow {
  font-size: 0.68rem !important; font-weight: 400 !important; letter-spacing: 0.24em !important;
  text-transform: uppercase !important; color: var(--uk-gray) !important; margin: 0 0 0.3rem !important;
}

/* ---- Labels ---- */
.gradio-container label span, .gradio-container .gr-form > div > label {
  font-size: 0.7rem !important; font-weight: 400 !important; letter-spacing: 0.14em !important;
  text-transform: uppercase !important; color: var(--uk-gray) !important;
}

/* ---- Inputs ---- */
.gradio-container input[type=text], .gradio-container textarea,
.gradio-container input[type=number], .gradio-container .gr-text-input {
  background: var(--uk-darkgray) !important;
  border: 1px solid #2a2a2a !important;
  color: var(--uk-cream) !important;
  font-family: 'Outfit', sans-serif !important;
  font-weight: 300 !important;
  border-radius: 6px !important;
}
.gradio-container input:focus, .gradio-container textarea:focus {
  border-color: var(--uk-red) !important;
  box-shadow: 0 0 0 1px var(--uk-red) !important;
}

/* ---- Radio / dropdown ---- */
.gradio-container .wrap.svelte-1p9xokt, .gradio-container [data-testid="block-info"] { color: var(--uk-gray) !important; }
.gradio-container input[type=radio] + span, .gradio-container .gr-check-radio label {
  color: var(--uk-cream) !important; font-weight: 300 !important;
}

/* ---- Sliders ---- */
.gradio-container input[type=range] { accent-color: var(--uk-red) !important; }

/* ---- Buttons ---- */
.gradio-container button.primary, .gradio-container .gr-button-primary {
  background: var(--uk-red) !important;
  color: var(--uk-white) !important;
  border: none !important;
  font-weight: 400 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  border-radius: 6px !important;
  transition: background 0.3s ease, transform 0.3s ease !important;
}
.gradio-container button.primary:hover { background: #c92d39 !important; transform: translateY(-1px); }
.gradio-container button.secondary, .gradio-container .gr-button-secondary {
  background: transparent !important;
  color: var(--uk-cream) !important;
  border: 1px solid var(--uk-gray) !important;
  font-weight: 400 !important;
  letter-spacing: 0.12em !important;
  text-transform: uppercase !important;
  border-radius: 6px !important;
  transition: all 0.3s ease !important;
}
.gradio-container button.secondary:hover { border-color: var(--uk-red) !important; color: var(--uk-red) !important; }
.gradio-container button.danger, .gradio-container .gr-button-danger {
  background: transparent !important;
  color: var(--uk-red) !important;
  border: 1px solid var(--uk-red) !important;
  font-weight: 400 !important;
  border-radius: 6px !important;
  transition: all 0.3s ease !important;
}
.gradio-container button.danger:hover { background: var(--uk-red) !important; color: var(--uk-white) !important; }

/* ---- Accordion ---- */
.gradio-container .label-wrap, .gradio-container .accordion > button {
  color: var(--uk-gray) !important; font-weight: 400 !important;
  letter-spacing: 0.16em !important; text-transform: uppercase !important; font-size: 0.7rem !important;
}

/* ---- Columns ---- */
#uk-left, #uk-right { padding-right: 0.6rem; }

/* ---- Output gallery ---- */
#uk-gallery .sfx-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 0.8rem; }
#uk-gallery .sfx-card { background: var(--uk-darkgray); border: 1px solid #2a2a2a; border-radius: 10px; padding: 0.7rem; display: flex; flex-direction: column; gap: 0.5rem; }
#uk-gallery .sfx-idx { font-size: 0.68rem; color: var(--uk-gray); letter-spacing: 0.12em; }
#uk-gallery .sfx-audio { width: 100%; height: 34px; }
#uk-gallery .sfx-row { display: flex; justify-content: space-between; align-items: baseline; gap: 0.4rem; }
#uk-gallery .sfx-name { font-size: 0.7rem; color: var(--uk-cream); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
#uk-gallery .sfx-size { font-size: 0.66rem; color: var(--uk-gray); flex-shrink: 0; }
#uk-gallery .sfx-dl { font-size: 0.7rem; color: var(--uk-red); text-decoration: none; text-align: center; border: 1px solid var(--uk-red); border-radius: 6px; padding: 0.28rem; transition: all 0.15s; }
#uk-gallery .sfx-dl:hover { background: var(--uk-red); color: var(--uk-white); }
#uk-gallery .sfx-empty { color: var(--uk-gray); font-size: 0.85rem; padding: 1rem 0; }
#uk-left::-webkit-scrollbar, #uk-right::-webkit-scrollbar { width: 6px; }
#uk-left::-webkit-scrollbar-thumb, #uk-right::-webkit-scrollbar-thumb { background: var(--uk-darkgray); border-radius: 3px; }

/* ---- Audio ---- */
.gradio-container .audio-container, .gradio-container [data-testid="waveform"] {
  background: var(--uk-darkgray) !important; border-radius: 6px !important;
}

/* ---- Code panels ---- */
.gradio-container .cm-editor, .gradio-container pre, .gradio-container code {
  background: var(--uk-darkgray) !important; border-radius: 6px !important;
  font-size: 0.78rem !important;
}
.gradio-container .tab-nav button { color: var(--uk-gray) !important; font-weight: 400 !important; letter-spacing: 0.1em; }
.gradio-container .tab-nav button.selected { color: var(--uk-red) !important; border-bottom-color: var(--uk-red) !important; }

/* ---- History table ---- */
#uk-history { color: var(--uk-cream) !important; font-size: 0.82rem !important; }
#uk-history td { padding: 0.35rem 0.6rem !important; border-bottom: 1px solid var(--uk-darkgray) !important; }
#uk-history td:first-child { color: var(--uk-red) !important; font-weight: 500; width: 2em; }
"""


def build_ui():
    with gr.Blocks(title="Woosh Studio", css=CSS) as demo:

        # ---- Header ----
        gr.HTML(
            """
            <div id="uk-header">
              <p class="eyebrow">Sony AI · Text-to-Audio · Jetson Orin</p>
              <h1>Woosh<span class="dot">.</span> Studio</h1>
              <p class="access">Access from anywhere on the tailnet at
                <code>http://jetson-orin.heron-minor.ts.net:7860</code></p>
            </div>
            """
        )

        # ---- Body: two columns filling the viewport ----
        with gr.Row(equal_height=False):
            # ---------- Left: controls ----------
            with gr.Column(scale=5, elem_id="uk-left"):
                gr.HTML('<p class="uk-eyebrow">Model</p>')
                model_key = gr.Radio(
                    ["DFlow (fast, distilled)", "Flow (full quality)"],
                    value="DFlow (fast, distilled)",
                    label="",
                    info="DFlow — 4 steps, ~4s, fast iteration · Flow — 50 steps, ~2-4 min, production",
                )

                prompt = gr.Textbox(
                    label="Prompt",
                    placeholder="e.g. glass shattering, footsteps on gravel, door slamming",
                    lines=2,
                )

                with gr.Row():
                    batch_size = gr.Slider(1, 4, value=1, step=1, label="Batch size")
                    cfg_scale = gr.Slider(0.0, 15.0, value=4.5, step=0.1, label="CFG scale")

                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Number(value=-1, label="Seed (-1 = random)", precision=0)
                    solver = gr.Dropdown(
                        ["dopri5", "dopri8", "bosh3", "adaptive_heun"],
                        value="dopri5",
                        label="ODE solver (Flow only)",
                    )
                    with gr.Row():
                        atol = gr.Number(value=0.001, label="Atol")
                        rtol = gr.Number(value=0.001, label="Rtol")

                with gr.Accordion("Export Settings", open=False):
                    output_format = gr.Radio(
                        ["ogg", "mp3", "wav"],
                        value="ogg",
                        label="Export format",
                    )
                    with gr.Row():
                        sample_rate = gr.Dropdown(
                            [("8 kHz", 8000), ("11 kHz", 11025), ("22 kHz", 22050),
                             ("44.1 kHz", 44100), ("48 kHz", 48000), ("96 kHz", 96000)],
                            value=48000, label="Sample rate",
                        )
                        bitrate = gr.Dropdown(
                            ["64k", "96k", "128k", "192k", "256k", "320k"],
                            value="256k", label="Bitrate",
                        )

                with gr.Row():
                    btn = gr.Button("Generate", variant="primary", size="lg")
                    push_btn = gr.Button("Save & Push", variant="secondary")
                    clear_btn = gr.Button("Clear All", variant="danger")

            # ---------- Right: output + history + API ----------
            with gr.Column(scale=4, elem_id="uk-right"):
                gr.HTML('<p class="uk-eyebrow">Output</p>')
                output_gallery = gr.HTML(
                    value='<div class="sfx-empty">No samples generated yet.</div>',
                    elem_id="uk-gallery",
                )
                info_text = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2,
                )

                gr.HTML('<p class="uk-eyebrow" style="margin-top:1.2rem;">Generation History</p>')
                history_display = gr.Markdown(value="**0 samples** — no generations yet", elem_id="uk-history")

                gr.HTML('<p class="uk-eyebrow" style="margin-top:1.2rem;">API & Integration</p>')
                with gr.Tabs():
                    with gr.Tab("CLI"):
                        gr.Code(value=SAMPLE_CLI, language="shell", label="")
                    with gr.Tab("JavaScript"):
                        gr.Code(value=SAMPLE_JS, language="javascript", label="")
                    with gr.Tab("Python"):
                        gr.Code(value=SAMPLE_PY, language="python", label="")
                    with gr.Tab("MCP"):
                        gr.Code(value=SAMPLE_MCP, language="json", label="")

        # ---- State ----
        history = gr.State([])  # list of generation dicts

        def gen_and_track(prompt, model_key, batch_size, cfg_scale, seed, solver, atol, rtol,
                          output_format, sample_rate, bitrate, current_history):
            results, elapsed, steps, seed_val, base_name = generate(
                prompt, model_key, batch_size, cfg_scale,
                seed, solver, atol, rtol,
            )

            # Encode every batch item into the chosen export format / sample-rate /
            # bitrate so the download links respect the Export Settings.
            # Exports go in a subdir so a "wav" export never overwrites its source wav.
            export_dir = os.path.join(OUTPUT_DIR, "export")
            os.makedirs(export_dir, exist_ok=True)
            download_paths = []
            for i in range(len(results)):
                wav_path = os.path.join(OUTPUT_DIR, f"{base_name}_{i}.wav")
                out_path = os.path.join(export_dir, f"{base_name}_{i}.{output_format}")
                if encode_audio(wav_path, out_path, output_format, int(sample_rate), bitrate):
                    download_paths.append(out_path)
                else:
                    download_paths.append(wav_path)  # fall back to raw wav

            entry = {
                "base_name": base_name,
                "prompt": prompt,
                "model": model_key,
                "elapsed": round(elapsed, 1),
                "seed": seed_val,
            }
            # Append to history
            current_history.append(entry)
            new_history = list(current_history)

            # Update history display
            lines = [f"**{len(new_history)} sample(s)**"]
            for i, e in enumerate(new_history, 1):
                lines.append(f"{i}. `{e['model'].split(' ')[0]}` — {e['prompt'][:45]} — {e['elapsed']}s")

            return (
                build_gallery_html(download_paths),  # one card per batch item
                new_history,
                "\n".join(lines),
                f"⚡ {elapsed:.1f}s · {len(results)} sample(s) · {output_format} · seed {seed_val}",
            )

        btn.click(
            fn=gen_and_track,
            inputs=[prompt, model_key, batch_size, cfg_scale, seed, solver, atol, rtol,
                    output_format, sample_rate, bitrate, history],
            outputs=[output_gallery, history, history_display, info_text],
            show_progress_on=output_gallery,
            api_name="generate",
        )

        def push_all_fn(history, output_format, bitrate, sample_rate):
            return push_all(history, output_format, bitrate, sample_rate)

        push_btn.click(
            fn=push_all_fn,
            inputs=[history, output_format, bitrate, sample_rate],
            outputs=info_text,
            api_name=False,  # UI-only; not an MCP/REST tool (does a git push)
        )

        def clear_h(history):
            return (
                '<div class="sfx-empty">No samples generated yet.</div>',
                [], "**0 samples** — no generations yet", "Cleared.",
            )

        clear_btn.click(
            fn=clear_h,
            inputs=[history],
            outputs=[output_gallery, history, history_display, info_text],
            api_name=False,  # UI-only helper
        )

        # Clean MCP tool: text prompt -> generated sound URL(s). Plain typed args,
        # no UI state, so it presents well to MCP clients.
        gr.api(generate_sound, api_name="generate_sound")

    return demo

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Woosh Unified Web UI")
    parser.add_argument("--server-name", default="0.0.0.0")
    parser.add_argument("--server-port", type=int, default=7860)
    args = parser.parse_args()

    # Singleton guard: refuse to start a second instance. Any duplicate launch
    # (e.g. an agent or supervisor relaunching us) takes the lock, fails the
    # non-blocking acquire, and exits immediately — no port fight, no double
    # model load. The lock auto-releases when this process dies.
    import fcntl
    global _SINGLETON_LOCK_FH
    _SINGLETON_LOCK_FH = open("/tmp/woosh_ui.singleton.lock", "w")
    try:
        fcntl.flock(_SINGLETON_LOCK_FH, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("Woosh UI already running (singleton lock held); exiting.")
        sys.exit(0)
    _SINGLETON_LOCK_FH.write(str(os.getpid()))
    _SINGLETON_LOCK_FH.flush()

    print(f"Loading models on {state.device}...")
    state.switch_model("DFlow")
    state.switch_model("Flow")
    print("Models ready.")

    demo = build_ui()
    demo.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        show_error=True,
        share=False,
        mcp_server=True,
        css=CSS,
        theme=gr.themes.Base(),
        allowed_paths=[OUTPUT_DIR],
    )

if __name__ == "__main__":
    main()
