
[Abdelrahman Shaker](https://amshaker.github.io/), [Muhammad Maaz](https://www.muhammadmaaz.com), [Chenhui Gou](https://scholar.google.com/citations?user=tlhShPsAAAAJ&hl=en), [Hamid Rezatofighi](https://scholar.google.com/citations?user=VxAuxMwAAAAJ&hl=en), [Salman Khan](https://salman-h-khan.github.io/), and [Fahad Khan](https://sites.google.com/view/fahadkhans/home)


<br>

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; [Paper](https://ieeexplore.ieee.org/document/10526382): [<img height="25" src="docs/images/mobile_videogpt_paper.png" width="25" />](https://ieeexplore.ieee.org/document/10526382)  , [Code](https://github.com/Amshaker/Mobile-VideoGPT): [<img height="25" src="docs/images/github.png" width="25" />](https://github.com/Amshaker/Mobile-VideoGPT), [Models](https://huggingface.co/collections/Abdelrahman-shaker/mobile-videogpt-fast-and-accurate-video-understanding-langu-67dc745074f8dd68d93b6b92): [<img height="25" src="docs/images/hf-logo.png" width="25" />](https://huggingface.co/collections/Abdelrahman-shaker/mobile-videogpt-fast-and-accurate-video-understanding-langu-67dc745074f8dd68d93b6b92), [Presentation](https://mbzuaiac-my.sharepoint.com/:b:/g/personal/abdelrahman_youssief_mbzuai_ac_ae/Eb5VOLUzoB5ChK_h21kWQQABXIWBqWWprqTZ2UzpwxQ50g?e=awUxc4): [<img height="25" src="docs/images/ppt.png" width="25" />](https://mbzuaiac-my.sharepoint.com/:b:/g/personal/abdelrahman_youssief_mbzuai_ac_ae/Eb5VOLUzoB5ChK_h21kWQQABXIWBqWWprqTZ2UzpwxQ50g?e=awUxc4)

<br>


## Mobile-VideoGPT Overview

<p align="center">
  <iframe width="600" height="338" src="https://www.youtube.com/embed/6Ueqq_D_mR0" 
          frameborder="0" allowfullscreen></iframe>
</p>


## Overall Architecture of Mobile-VideoGPT

We propose Mobile-VideoGPT, an efficient multimodal framework designed to operate with fewer than a billion parameters. Unlike traditional video large multimodal models, Mobile-VideoGPT consists of lightweight dual visual encoders, efficient projectors, and a small language model (SLM) with real-time throughput. To further improve efficiency, we present an Attention-Based Frame Scoring mechanism to select the key-frames, along with an efficient token projector that prunes redundant visual tokens and preserves essential contextual cues. We evaluate our model across well-established six video understanding benchmarks (e.g., MVBench, EgoSchema, NextQA, and PerceptionTest), and our results show that Mobile-VideoGPT-0.5B can generate up to 46 tokens per second while outperforming existing state-of-the-art 0.5B-parameter competitors.

<p align="center">
  <img src="docs/images/method_figure_a.png" alt="Mobile-VideoGPT Architectural Overview" style="width: 600px; height: auto;">
</p>


---
## 📊 Evaluation Summary on 6 benchmarks:

<p align="center">
  <img src="docs/images/Intro_figure_2.png" alt="Contributions" style="width: 600px; height: auto;">
</p>

## 🛠️ Installation 

We recommend setting up a conda environment for the project:
```shell
conda create --name=mobile_videogpt python=3.11
conda activate mobile_videogpt

git clone https://github.com/Amshaker/Mobile-VideoGPT
cd Mobile-VideoGPT

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.41.0

pip install -r requirements.txt

export PYTHONPATH="./:$PYTHONPATH"
```
Install [VideoMamba](https://github.com/OpenGVLab/VideoMamba). VideoMamba is the efficient video encoder in our architecture. Clone and install it using the following commands:
```shell
git clone https://github.com/OpenGVLab/VideoMamba
cd VideoMamba
pip install -e causal-conv1d
pip install -e mamba
```
Additionally, install [FlashAttention](https://github.com/HazyResearch/flash-attention) for training,
```shell
pip install ninja

git clone https://github.com/HazyResearch/flash-attention.git
cd flash-attention
python setup.py install
```
---

## 🔥 Running Inference

The following script demonstrates how to load Mobile-VideoGPT-1.5B, tokenize the input prompt, and generate a response:

```python
import sys
import torch
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# Mobile-VideoGPT Imports
from mobilevideogpt.utils import preprocess_input

# Model and tokenizer paths
pretrained_path = "Mobile-VideoGPT-1.5B" 
video_path = "sample_videos/v_JspVuT6rsLA.mp4"
prompt = "Can you describe what is happening in the video in detail?"

# Load model and tokenizer
config = AutoConfig.from_pretrained(pretrained_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    pretrained_path,
    config=config,
    torch_dtype=torch.float16
).cuda()

# Preprocess input
input_ids, video_frames, context_frames, stop_str = preprocess_input(
    model, tokenizer, video_path, prompt
)

# Run inference
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=torch.stack(video_frames, dim=0).half().cuda(),
        context_images=torch.stack(context_frames, dim=0).half().cuda(),
        do_sample=False,
        temperature=0,
        top_p=1,
        num_beams=1,
        max_new_tokens=1024,
        use_cache=True,
    )
# Decode output
outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
if outputs.endswith(stop_str):
    outputs = outputs[:-len(stop_str)].strip()

print("🤖 Mobile-VideoGPT Output:", outputs)
```
✅ Expected Output:
```
🤖 Mobile-ViideoGPT Output:  In the video, a young boy is playing the violin in front of an adult who is playing the piano. The boy appears to be focused on his performance and is wearing a blue shirt with white stripes down the sleeves. He plays the violin with great concentration and skill, moving his fingers along the strings with precision. Meanwhile, the adult pianist sits behind him and plays the piano alongside him. Both musicians are dressed casually, suggesting that they may be practicing or performing for friends or family. The setting seems to be a cozy room with wooden flooring, giving it a warm and inviting atmosphere. Overall, this video showcases two talented musicians sharing a moment of musical collaboration, creating beautiful music together.
```
## 📊 Quantitative Evaluation:
We provide instructions on how to reproduce Mobile-VideoGPT-0.5B and Mobile-VideoGPT-1.5B results on MVBench, PerceptionTest, NextQA, MLVU, EgoSchema, and ActNet-QA. Please follow the instructions at [eval/README.md](eval/README.md).

### Benchmark Evaluation:
<p align="center">
  <img src="docs/images/overall_comparison.png" alt="All_benchmarks" style="width: 600px; height: auto;">
</p>

---
### Detailed Evaluation on MVBench:
<p align="center">
  <img src="docs/images/mvbench_comparison.png" alt="MVBench_quantitative" style="width: 600px; height: auto;">
</p>

---

## Training:
We provide unified scripts for pretraining and finetuning of Mobile-VideoGPT. Please follow the instructions at [scripts/README.md](scripts/README.md).

---
## Qualitative Examples:
Qualitative comparison between the proposed Mobile Video-GPT-0.5B, LLaVA-OneVision-0.5B, and LLaVa-Mini-8B. The output highlights both video comprehension quality and speed performance in terms of latency and throughput (tokens per second):

<p align="center">
  <img src="docs/images/qualitative_results_1-1.png" alt="Contributions" style="width: 600px; height: auto;">
</p>

---

Multi-turn conversation for Mobile-VideoGPT:
<p align="center">
  <img src="docs/images/qualitative_results_2-1.png" alt="Contributions" style="width: 600px; height: auto;">
</p>

---



## 📜 Citations:

If you're using Mobile-VideoGPT in your research or applications, please cite using this BibTeX:
```bibtex
@article{Shaker2025MobileVideoGPT,
    title={Mobile-VideoGPT: Fast and Accurate Video Understanding Language Model},
    author={Shaker, Abdelrahman and Maaz, Muhammad and Rezatofighi, Hamid and Khan, Salman and Khan, Fahad Shahbaz},
    journal={arxiv},
    year={2025},
    url={https://arxiv.org/abs/X.X}
}
```

## Contact
Should you have any questions, please create an issue on our GitHub repository or contact me at abdelrahman.youssief@mbzuai.ac.ae.

