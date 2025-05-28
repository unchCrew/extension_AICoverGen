# AICoverGen Extension

AICoverGen is a powerful, autonomous pipeline for generating AI-powered song covers using RVC v2 trained AI voices. Create covers from YouTube videos or local audio files with ease. Perfect for developers integrating singing functionality into AI assistants, chatbots, or VTubers, and for enthusiasts who want their favorite characters to sing their favorite songs.

**Showcase**: [Watch it in action](https://www.youtube.com/watch?v=2qZuE4WM7CM)  
**Setup Guide**: [Step-by-step video tutorial](https://www.youtube.com/watch?v=pdlhk4vVHQk)

![WebUI Generate](images/webui_generate.png)

The WebUI is actively developed and tested, available for both local and Google Colab environments. Try it now and bring your AI covers to life!

## Table of Contents

- [Features](#features)
- [Changelog](#changelog)
- [Setup](#setup)
  - [Install Git and Python](#install-git-and-python)
  - [Install FFmpeg and SoX](#install-ffmpeg-and-sox)
  - [Clone AICoverGen Repository](#clone-aicovergen-repository)
  - [Download Required Models](#download-required-models)
- [Usage with WebUI](#usage-with-webui)
  - [Download RVC Models via WebUI](#download-rvc-models-via-webui)
  - [Upload RVC Models via WebUI](#upload-rvc-models-via-webui)
  - [Running the Pipeline via WebUI](#running-the-pipeline-via-webui)
- [Usage with CLI](#usage-with-cli)
  - [Manual Download of RVC Models](#manual-download-of-rvc-models)
  - [Running the Pipeline via CLI](#running-the-pipeline-via-cli)
- [Colab Notebook](#colab-notebook)
- [Terms of Use](#terms-of-use)
- [Disclaimer](#disclaimer)

## Features

- **WebUI**: Intuitive interface for easy cover generation and voice model management.
- **Local and Cloud Support**: Run locally or on Google Colab for accessibility.
- **Flexible Input**: Generate covers from YouTube videos or local audio files.
- **Advanced Audio Controls**: Adjust pitch, volume, reverb, and more for vocals and instrumentals.
- **RVC v2 Model Support**: Use pre-trained or custom-trained RVC models.
- **High-Quality Output**: Supports rmvpe pitch extraction and multiple audio formats (WAV, MP3).
- **Customizable Pipeline**: Keep intermediate files (e.g., isolated vocals/instrumentals) for further editing.

## Changelog

- Added WebUI for streamlined conversions and voice model downloads.
- Support for local audio file inputs.
- Option to retain intermediate files (e.g., isolated vocals/instrumentals).
- Searchable table for public voice models with tag filters.
- Pixeldrain support for voice model downloads.
- New rmvpe pitch extraction for faster, higher-quality vocal conversions.
- Volume controls for main vocals, backup vocals, and instrumentals.
- Index rate for fine-tuned voice conversion.
- Reverb controls for main vocals.
- Local network sharing for WebUI.
- Advanced RVC options: filter radius, RMS mix rate, and breath protection.
- Local file uploads via file browser.
- Support for uploading locally trained RVC v2 models.
- Pitch detection method selection (e.g., rmvpe, mangio-crepe).
- Global pitch adjustment for vocals and instrumentals (like changing song key).
- Output format selection (WAV or MP3).

## Setup

### Install Git and Python

1. Install Git: Follow the [official guide](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git).
2. Install Python 3.9: Refer to this [guide](https://realpython.com/installing-python/). **Note**: Other Python versions may cause dependency conflicts.

### Install FFmpeg and SoX

1. Install FFmpeg: Follow this [tutorial](https://www.hostinger.com/tutorials/how-to-install-ffmpeg).
2. Install SoX: Follow this [guide](https://www.tutorialexample.com/a-step-guide-to-install-sox-sound-exchange-on-windows-10-python-tutorial/) and add SoX to your Windows PATH.

### Clone AICoverGen Repository

Clone the repository and install dependencies:

```bash
git clone https://github.com/unchCrew/extension_AICoverGen
cd extension_AICoverGen
pip install -r requirements.txt
```

### Download Required Models

Download the MDXNET vocal separation and Hubert base models:

```bash
python src/download_models.py
```

## Usage with WebUI

Launch the WebUI:

```bash
python src/webui.py
```

| Flag                  | Description |
|-----------------------|-------------|
| `--share`             | Creates a public URL (useful for Google Colab). |
| `--listen`            | Makes the WebUI accessible on your local network. |
| `--listen-host`       | Specifies the server hostname. |
| `--listen-port`       | Specifies the server port. |

Once you see `Running on local URL: http://127.0.0.1:7860`, click the link to access the WebUI.

### Download RVC Models via WebUI

![WebUI Download Model](images/webui_dl_model.png)

1. Navigate to the **Download Model** tab.
2. Paste the RVC model download link (e.g., from [AI Hub Discord](https://discord.gg/aihub)).
3. Provide a unique name for the model.
4. Click **Download**. The model (`.pth` and optional `.index` files) will be saved.
5. After seeing `[NAME] Model successfully downloaded!`, refresh the model list in the **Generate** tab.

### Upload RVC Models via WebUI

![WebUI Upload Model](images/webui_upload_model.png)

1. Navigate to the **Upload Model** tab.
2. Upload your locally trained RVC v2 model (`.pth` and optional `.index` files).
3. Follow the prompts to name and save the model.
4. After seeing `[NAME] Model successfully uploaded!`, refresh the model list in the **Generate** tab.

### Running the Pipeline via WebUI

![WebUI Generate](images/webui_generate.png)

1. Select a voice model from the **Voice Models** dropdown. Click **Update** if models were added manually to the `rvc_models` directory.
2. Enter a YouTube URL or local audio file path in the **Song Input** field.
3. Adjust the pitch (-12, 0, or 12) to match the original vocals and RVC model for optimal tuning.
4. Expand the advanced options to fine-tune voice conversion and audio mixing settings.
5. Click **Generate**. The AI cover will be ready in a few minutes, depending on your GPU.

## Usage with CLI

### Manual Download of RVC Models

1. Unzip the RVC model (if needed).
2. Place the `.pth` and optional `.index` files in a new folder under `rvc_models`.
3. Ensure each folder contains only one `.pth` and one `.index` file. Example structure:

The directory structure should look something like this:
```
├── rvc_models
│   ├── John
│   │   ├── JohnV2.pth
│   │   └── added_IVF2237_Flat_nprobe_1_v2.index
│   ├── May
│   │   ├── May.pth
│   │   └── added_IVF2237_Flat_nprobe_1_v2.index
│   ├── MODELS.txt
│   └── hubert_base.pt
├── mdxnet_models
├── song_output
└── src
 ```
### Running the Pipeline via CLI

Run the pipeline with:

```bash
python src/main.py -i SONG_INPUT -dir RVC_DIRNAME -p PITCH_CHANGE [options]
```

| Flag                          | Description |
|-------------------------------|-------------|
| `-i SONG_INPUT`               | YouTube URL or local audio file path (in quotes). |
| `-dir MODEL_DIR_NAME`         | Folder name in `rvc_models` with `.pth` and `.index` files. |
| `-p PITCH_CHANGE`             | Pitch adjustment for AI vocals (e.g., 1 for male-to-female, -1 for female-to-male). |
| `-k, --keep-files`            | Keep intermediate files (e.g., isolated vocals). |
| `-ir INDEX_RATE`              | Accent retention (0 to 1, default: 0.5). |
| `-fr FILTER_RADIUS`           | Median filtering for pitch (0 to 7, default: 3). |
| `-rms RMS_MIX_RATE`           | Original vs. fixed loudness (0 to 1, default: 0.25). |
| `-palgo PITCH_DETECTION_ALGO` | Pitch detection algorithm (default: rmvpe). |
| `-hop CREPE_HOP_LENGTH`       | Pitch check frequency for mangio-crepe (default: 128). |
| `-pro PROTECT`                | Breath/consonant retention (0 to 0.5, default: 0.33). |
| `-mv MAIN_VOCALS_VOLUME`      | Main vocals volume in decibels (default: 0). |
| `-bv BACKUP_VOCALS_VOLUME`    | Backup vocals volume in decibels (default: 0). |
| `-iv INSTRUMENTAL_VOLUME`     | Instrumental volume in decibels (default: 0). |
| `-pall PITCH_CHANGE_ALL`      | Global pitch shift in semitones (default: 0). |
| `-rsize REVERB_SIZE`          | Reverb room size (0 to 1, default: 0.15). |
| `-rwet REVERB_WETNESS`        | Reverb level for vocals (0 to 1, default: 0.2). |
| `-rdry REVERB_DRYNESS`        | Non-reverb vocal level (0 to 1, default: 0.8). |
| `-rdamp REVERB_DAMPING`       | High-frequency reverb absorption (0 to 1, default: 0.7). |
| `-oformat OUTPUT_FORMAT`      | Output format (wav or mp3, default: mp3). |

## Colab Notebook

Run AICoverGen on Google Colab for GPU-free processing:

- **WebUI Version**: [Open in Colab](https://colab.research.google.com/github/TheNeodev/Notebook/blob/main/AICoverGen_colab.ipynb)

- **NoUI Version**: Will added comming soon...

To update in Colab:
1. Click **Runtime** > **Disconnect and delete runtime**.
2. Follow the notebook instructions to restart.

## Terms of Use

The following uses of converted voices are prohibited:
- Criticizing or attacking individuals.
- Advocating for or against political positions, religions, or ideologies.
- Displaying explicit content without proper zoning.
- Selling voice models or generated clips.
- Impersonating voice owners with malicious intent.
- Engaging in fraud or identity theft.

## Disclaimer

The developer is not liable for any damages arising from the use or misuse of this software.
