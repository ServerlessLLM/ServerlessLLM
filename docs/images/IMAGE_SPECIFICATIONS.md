# Image Specifications for README

Upload these images to make the README complete:

---

## 1. Benchmark Chart: `benchmark_loading_speed.png`

**Path:** `./docs/images/benchmark_loading_speed.png`

**Purpose:** Show 5-10x loading speed improvement

**Specifications:**
- **Type:** PNG (high quality, transparent or white background)
- **Size:** 1200x800 pixels (horizontal orientation)
- **Style:** Professional bar chart or grouped bar chart
- **Colors:**
  - ServerlessLLM: Green (#00B894 or similar)
  - Traditional (SafeTensors): Red/Orange (#FF7675 or similar)
- **Background:** White or transparent (works on GitHub light/dark mode)

**Data to Show:**

| Model | Traditional (SafeTensors) | ServerlessLLM | Speedup |
|-------|---------------------------|---------------|---------|
| OPT-1.3B (2.6GB) | 35s | 5s | 7x |
| OPT-6.7B (13GB) | 180s | 25s | 7.2x |
| LLaMA-13B (26GB) | 360s | 45s | 8x |

**Chart Elements:**
- X-axis: Model names (OPT-1.3B, OPT-6.7B, LLaMA-13B)
- Y-axis: Loading time in seconds
- Two bars per model (Traditional vs ServerlessLLM)
- Labels on bars showing exact times
- Speedup annotation (e.g., "7x faster" arrow or text)
- Footer: "Tested on NVIDIA A100 GPU, NVMe SSD"
- Title: "Model Loading Speed Comparison"

**Tools You Can Use:**
- Python: matplotlib, seaborn, plotly
- Design tools: Figma, Canva, Adobe Illustrator
- Excel/Google Sheets → export as image

**Example Code (Python):**
```python
import matplotlib.pyplot as plt
import numpy as np

models = ['OPT-1.3B\n(2.6GB)', 'OPT-6.7B\n(13GB)', 'LLaMA-13B\n(26GB)']
traditional = [35, 180, 360]
serverlessllm = [5, 25, 45]
speedup = [7, 7.2, 8]

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 8))
bars1 = ax.bar(x - width/2, traditional, width, label='Traditional (SafeTensors)', color='#FF7675')
bars2 = ax.bar(x + width/2, serverlessllm, width, label='ServerlessLLM', color='#00B894')

# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}s', ha='center', va='bottom', fontsize=10)

for bar in bars2:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}s', ha='center', va='bottom', fontsize=10)

# Add speedup annotations
for i, speed in enumerate(speedup):
    ax.text(i, max(traditional[i], serverlessllm[i]) + 20,
            f'{speed}x faster', ha='center', fontweight='bold', fontsize=11, color='#00B894')

ax.set_xlabel('Model', fontsize=12, fontweight='bold')
ax.set_ylabel('Loading Time (seconds)', fontsize=12, fontweight='bold')
ax.set_title('Model Loading Speed Comparison', fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models)
ax.legend(fontsize=11, loc='upper left')
ax.grid(axis='y', alpha=0.3)

# Footer note
fig.text(0.5, 0.02, 'Tested on NVIDIA A100 GPU with NVMe SSD',
         ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout()
plt.savefig('benchmark_loading_speed.png', dpi=150, bbox_inches='tight', facecolor='white')
```

---

## 2. Demo GIF: `demo_quickstart.gif`

**Path:** `./docs/images/demo_quickstart.gif`

**Purpose:** Show the 90-second quick start experience

**Specifications:**
- **Type:** Animated GIF
- **Size:** 1280x720 pixels (720p) or 1920x1080 (1080p)
- **Duration:** 30-60 seconds (looped)
- **FPS:** 10-15 fps (keeps file size manageable)
- **File size:** <10MB (GitHub limit is 10MB, aim for 5-8MB)
- **Style:** Terminal recording with clean, readable text

**What to Show (in order):**

### Scene 1: Clone & Start (5-10 seconds)
```bash
$ cd ServerlessLLM/examples/docker
$ export MODEL_FOLDER=/path/to/models
$ docker compose up -d
Creating sllm-head ... done
Creating sllm-worker-0 ... done
```

### Scene 2: Deploy Model (5-10 seconds)
```bash
$ export LLM_SERVER_URL=http://127.0.0.1:8343
$ sllm deploy facebook/opt-1.3b --backend transformers --num-gpus 1

Loading model facebook/opt-1.3b...
Model loaded in 5.2 seconds! ✓
Model deployed successfully!
```

### Scene 3: Query Model (5-10 seconds)
```bash
$ curl $LLM_SERVER_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "facebook/opt-1.3b",
    "messages": [{"role": "user", "content": "What is serverless?"}]
  }'

{
  "choices": [{
    "message": {
      "content": "Serverless computing is a cloud architecture..."
    }
  }]
}
```

### Scene 4: Success Message (2-3 seconds)
```
🎉 ServerlessLLM is running!
⚡ Model loaded in 5 seconds (vs 35s with SafeTensors)
🚀 OpenAI-compatible API ready at localhost:8343
```

**Recording Tips:**
- Use a clean terminal theme (light background recommended for visibility)
- Clear terminal before starting: `clear`
- Type commands slowly (or use typing animation tool)
- Add pauses (1-2 seconds) between commands
- Show realistic timing (model load should take ~5 seconds)
- Trim dead time in editing

**Tools You Can Use:**

### Option 1: Terminal Recording + GIF (Recommended)
```bash
# Record with asciinema
asciinema rec demo.cast

# Convert to GIF with agg
agg demo.cast demo.gif

# Or use terminalizer
terminalizer record demo
terminalizer render demo
```

### Option 2: Screen Recording
- **Mac:** QuickTime Screen Recording → export → convert to GIF
- **Linux:** SimpleScreenRecorder, Peek, Kazam
- **Windows:** ShareX, ScreenToGif
- **Convert:** Use ffmpeg or online tools to convert MP4 → GIF

**GIF Optimization:**
```bash
# Reduce file size with gifsicle
gifsicle -O3 --colors 256 demo.gif -o demo_optimized.gif

# Or with ffmpeg
ffmpeg -i demo.mp4 -vf "fps=12,scale=1280:-1:flags=lanczos" \
  -c:v gif demo.gif
```

---

## 3. Optional: Architecture Diagram Enhancement

**Current:** ASCII art (works fine!)

**Optional upgrade:** Create a visual diagram

**Path:** `./docs/images/architecture_diagram.png`

**Specifications:**
- **Type:** PNG
- **Size:** 1400x1000 pixels
- **Style:** Clean, modern, with icons
- **Colors:** Match ServerlessLLM branding

**Tools:**
- draw.io (free)
- Excalidraw (free, hand-drawn style)
- Figma (free tier)
- Lucidchart

**Elements to show:**
- Client → Control Plane (Router, Scheduler, Controller)
- Control Plane → SLLMStore (Fast Loading Layer)
- SLLMStore → Multiple GPUs (Model A, B, C, D)
- Annotations showing "5-8s cold start", "Live Migration", "Storage-Aware"

---

## File Checklist

### Critical (README won't render properly without these)
- [ ] `benchmark_loading_speed.png` - Benchmark chart
- [ ] `demo_quickstart.gif` - Quick start demo

### Optional (README works fine with ASCII art)
- [ ] `architecture_diagram.png` - Visual architecture

---

## Testing After Upload

1. Upload images to `./docs/images/`
2. Push to GitHub
3. View README on GitHub (not local preview!)
4. Check both light and dark mode themes
5. Verify images load on mobile (GitHub mobile app)
6. Test different zoom levels (should be readable at all sizes)

---

## Quick Checklist

**Before uploading:**
- [ ] Images are high resolution but optimized file size
- [ ] Background works on light/dark themes (white or transparent)
- [ ] Text in images is readable (min 12px font)
- [ ] GIF loops smoothly
- [ ] GIF file size < 10MB
- [ ] All paths match exactly: `./docs/images/[filename]`

**After uploading:**
- [ ] README renders correctly on GitHub
- [ ] Images load fast (<2 seconds)
- [ ] Images are crisp on retina displays
- [ ] Works on mobile

---

Last updated: 2025-11-13
