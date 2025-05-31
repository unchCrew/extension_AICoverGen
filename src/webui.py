import json
import os
import shutil
import urllib.request
import zipfile
from argparse import ArgumentParser
from typing import List, Tuple, Optional, Dict
import gradio as gr
from src.main import song_cover_pipeline

BASE_DIR = os.getcwd()
sys.path.append(now_dir)

MDXNET_MODELS_DIR = os.path.join(BASE_DIR, "mdxnet_models")
RVC_MODELS_DIR = os.path.join(BASE_DIR, "rvc_models")
OUTPUT_DIR = os.path.join(BASE_DIR, "song_output")
PUBLIC_MODELS_FILE = os.path.join(RVC_MODELS_DIR, "public_models.json")
EXCLUDED_MODELS = {"hubert_base.pt", "MODELS.txt", "public_models.json", "rmvpe.pt"}
DEFAULT_MODEL_SETTINGS = {
    "pitch": 0,
    "main_gain": 0,
    "backup_gain": 0,
    "inst_gain": 0,
    "index_rate": 0.5,
    "filter_radius": 3,
    "rms_mix_rate": 0.25,
    "protect": 0.33,
    "f0_method": "rmvpe",
    "crepe_hop_length": 128,
    "pitch_all": 0,
    "reverb_rm_size": 0.15,
    "reverb_wet": 0.2,
    "reverb_dry": 0.8,
    "reverb_damping": 0.7,
    "output_format": "mp3",
}

def get_current_models(models_dir: str) -> List[str]:
    """Retrieve list of available models, excluding specified files."""
    try:
        return [item for item in os.listdir(models_dir) if item not in EXCLUDED_MODELS]
    except FileNotFoundError:
        raise gr.Error(f"Models directory {models_dir} not found.")

def load_public_models_data() -> dict:
    """Load public models from JSON file."""
    try:
        with open(PUBLIC_MODELS_FILE, encoding="utf-8") as infile:
            return json.load(infile)
    except FileNotFoundError:
        raise gr.Error(f"Public models file {PUBLIC_MODELS_FILE} not found.")
    except json.JSONDecodeError:
        raise gr.Error("Invalid public models JSON format.")

def extract_zip(extraction_folder: str, zip_path: str, progress: gr.Progress = gr.Progress()) -> Tuple[Optional[str], Optional[str]]:
    """Extract zip file and find model and index files."""
    os.makedirs(extraction_folder, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_path)

    index_filepath, model_filepath = None, None
    for root, _, files in os.walk(extraction_folder):
        for name in files:
            file_path = os.path.join(root, name)
            if name.endswith(".index") and os.path.getsize(file_path) > 1024 * 100:
                index_filepath = file_path
            if name.endswith(".pth") and os.path.getsize(file_path) > 1024 * 1024 * 40:
                model_filepath = file_path

    if not model_filepath:
        raise gr.Error(f"No valid .pth model file found in {extraction_folder}.")

    # Move files to extraction folder root
    if model_filepath:
        shutil.move(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        shutil.move(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    # Clean up subdirectories
    for item in os.listdir(extraction_folder):
        item_path = os.path.join(extraction_folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)

    return model_filepath, index_filepath

def download_online_model(url: str, dir_name: str, progress: gr.Progress = gr.Progress()) -> gr.Dropdown:
    """Download and extract a model from a URL."""
    if not url or not dir_name:
        raise gr.Error("URL and model name are required.")
    
    extraction_folder = os.path.join(RVC_MODELS_DIR, dir_name)
    if os.path.exists(extraction_folder):
        raise gr.Error(f"Voice model directory {dir_name} already exists!")

    try:
        zip_name = url.split("/")[-1]
        zip_path = zip_name
        if "pixeldrain.com" in url:
            url = f"https://pixeldrain.com/api/file/{zip_name}"

        progress(0, desc=f"Downloading voice model: {dir_name}")
        urllib.request.urlretrieve(url, zip_path, reporthook=lambda b, bsize, tsize: progress(b * bsize / tsize))
        progress(0.5, desc="Extracting zip...")
        extract_zip(extraction_folder, zip_path)
        gr.Info(f"Model {dir_name} successfully downloaded!")
        return gr.Dropdown(choices=get_current_models(RVC_MODELS_DIR))
    except Exception as e:
        raise gr.Error(f"Download failed: {str(e)}")

def upload_local_model(zip_path: str, dir_name: str, progress: gr.Progress = gr.Progress()) -> gr.Dropdown:
    """Upload and extract a local model zip file."""
    if not zip_path or not dir_name:
        raise gr.Error("Zip file and model name are required.")

    extraction_folder = os.path.join(RVC_MODELS_DIR, dir_name)
    if os.path.exists(extraction_folder):
        raise gr.Error(f"Voice model directory {dir_name} already exists!")

    try:
        progress(0.5, desc="Extracting zip...")
        extract_zip(extraction_folder, zip_path)
        gr.Info(f"Model {dir_name} successfully uploaded!")
        return gr.Dropdown(choices=get_current_models(RVC_MODELS_DIR))
    except Exception as e:
        raise gr.Error(f"Upload failed: {str(e)}")

def filter_available_public_models(public_models: Dict, local_models: List[str]) -> Tuple[List[List[str]], List[str]]:
    """
    Filter public models to exclude locally installed models and prepare table data.

    Args:
        public_models: Dictionary containing public models data, with a "voice_models" key.
        local_models: List of locally installed model names.

    Returns:
        Tuple containing:
        - List of model data [name, description, credit, url, tags] for display.
        - List of available tags.

    Raises:
        KeyError: If 'voice_models' or required model fields are missing.
        TypeError: If input types are incorrect.
    """
    if not isinstance(public_models, dict) or "voice_models" not in public_models:
        raise gr.Error("Invalid public models data: 'voice_models' key missing.")
    if not isinstance(local_models, list):
        raise TypeError("local_models must be a list of strings.")

    local_models_set = set(local_models)
    models_table = []
    
    for model in public_models["voice_models"]:
        if not all(key in model for key in ["name", "description", "credit", "url", "tags"]):
            continue
        if model["name"] not in local_models_set:
            models_table.append([
                model["name"],
                model["description"],
                model["credit"],
                model["url"],
                ", ".join(model["tags"])
            ])

    tags = list(public_models.get("tags", {}).keys())
    return models_table, tags

def filter_models(tags: List[str], query: str, public_models: dict) -> List[List[str]]:
    """Filter public models based on tags and search query."""
    models_table = []
    for model in public_models["voice_models"]:
        model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
        if (not tags or all(tag in model["tags"] for tag in tags)) and (not query or query.lower() in model_attributes):
            models_table.append([model["name"], model["description"], model["credit"], model["url"], ", ".join(model["tags"])])
    return models_table

def create_generate_tab(voice_models: List[str], visibility_state: gr.State) -> gr.Blocks:
    """Create the Generate tab for the Gradio UI."""
    with gr.Tab("Generate"):
        with gr.Group(elem_classes=["section"]):
            gr.Markdown("### Main Options")
            with gr.Row(variant="compact", equal_height=True):
                rvc_model = gr.Dropdown(voice_models, label="Voice Models", info="Add models to rvc_models and refresh.")
                ref_btn = gr.Button("🔄 Refresh Models", variant="primary")

            with gr.Row(variant="compact", equal_height=True):
                with gr.Column(visible=visibility_state) as yt_link_col:
                    song_input = gr.Textbox(label="Song Input", placeholder="YouTube link or local file path")
                    show_file_upload_button = gr.Button("Upload File Instead")
                with gr.Column(visible=not visibility_state) as file_upload_col:
                    local_file = gr.File(label="Audio File", type="filepath")
                    song_input_file = gr.UploadButton("📂 Upload Audio", file_types=["audio"], variant="primary")
                    show_yt_link_button = gr.Button("Use YouTube Link/Path Instead")
                with gr.Column():
                    pitch = gr.Slider(-3, 3, value=0, step=1, label="Vocal Pitch (Octaves)", info="1 for male-to-female, -1 for vice-versa")
                    pitch_all = gr.Slider(-12, 12, value=0, step=1, label="Overall Pitch (Semitones)", info="Adjusts vocals and instrumentals")

            show_file_upload_button.click(lambda state: (not state, state, "", None), inputs=[visibility_state], outputs=[yt_link_col, file_upload_col, song_input, local_file])
            show_yt_link_button.click(lambda state: (not state, state, "", None), inputs=[visibility_state], outputs=[yt_link_col, file_upload_col, song_input, local_file])
            song_input_file.upload(lambda file: (file.name, file.name), inputs=[song_input_file], outputs=[local_file, song_input])

        with gr.Accordion("Voice Conversion Options", open=False):
            with gr.Row(variant="compact"):
                index_rate = gr.Slider(0, 1, value=0.5, label="Index Rate", info="AI voice accent retention")
                filter_radius = gr.Slider(0, 7, value=3, step=1, label="Filter Radius", info="Reduces breathiness if >=3")
                rms_mix_rate = gr.Slider(0, 1, value=0.25, label="RMS Mix Rate", info="Original (0) vs. fixed loudness (1)")
                protect = gr.Slider(0, 0.5, value=0.33, label="Protect Rate", info="Protects consonants/breath sounds")
                f0_method = gr.Radio(choices=["pm", "mangio-crepe", "crepe-full", "crepe-tiny", "rmvpe" "pyin"], value="rmvpe", label="Pitch Detection", info="rmvpe for clarity, mangio-crepe for smoothness")
                crepe_hop_length = gr.Slider(32, 320, value=128, step=1, visible=False, label="Crepe Hop Length", info="Lower values improve pitch but risk cracks")
            f0_method.change(lambda algo: gr.Slider(visible=algo == "mangio-crepe"), inputs=f0_method, outputs=crepe_hop_length)
            keep_files = gr.Checkbox(label="Keep Intermediate Files", value=False)

        with gr.Accordion("Audio Mixing Options", open=False):
            gr.Markdown("### Volume (dB)")
            with gr.Row(variant="compact"):
                main_gain = gr.Slider(-20, 20, value=0, step=1, label="Main Vocals")
                backup_gain = gr.Slider(-20, 20, value=0, step=1, label="Backup Vocals")
                inst_gain = gr.Slider(-20, 20, value=0, step=1, label="Music")
            gr.Markdown("### Reverb (AI Vocals)")
            with gr.Row(variant="compact"):
                reverb_rm_size = gr.Slider(0, 1, value=0.15, label="Room Size")
                reverb_wet = gr.Slider(0, 1, value=0.2, label="Wetness")
                reverb_dry = gr.Slider(0, 1, value=0.8, label="Dryness")
                reverb_damping = gr.Slider(0, 1, value=0.7, label="Damping")
            output_format = gr.Dropdown(["mp3", "wav"], value="mp3", label="Output Format", info="mp3: smaller, wav: higher quality")

        clear_btn = gr.Button("Clear", variant="secondary")
        with gr.Row(equal_height=True):    
            generate_btn = gr.Button("Generate", variant="primary")
            ai_cover = gr.Audio(label="AI Cover", interactive=False)

        ref_btn.click(lambda: gr.Dropdown(choices=get_current_models(RVC_MODELS_DIR)), outputs=rvc_model)
        is_webui = gr.State(value=1)
        generate_btn.click(
            song_cover_pipeline,
            inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain, inst_gain,
                    index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length, protect, pitch_all,
                    reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, output_format],
            outputs=[ai_cover]
        )
        clear_btn.click(
            lambda: [DEFAULT_MODEL_SETTINGS[key] for key in ["pitch", "main_gain", "backup_gain", "inst_gain", "index_rate",
                                                             "filter_radius", "rms_mix_rate", "protect", "f0_method",
                                                             "crepe_hop_length", "pitch_all", "reverb_rm_size", "reverb_wet",
                                                             "reverb_dry", "reverb_damping", "output_format"]] + [None, ""],
            outputs=[pitch, main_gain, backup_gain, inst_gain, index_rate, filter_radius, rms_mix_rate, protect,
                     f0_method, crepe_hop_length, pitch_all, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping,
                     output_format, ai_cover, song_input]
        )
    return rvc_model, ai_cover

def create_download_model_tab(voice_models: List[str], public_models: dict) -> None:
    """Create the Download Model tab for the Gradio UI."""
    with gr.Tab("Download Model"):
        with gr.Tabs():
            with gr.Tab("From URL"):
                model_zip_link = gr.Textbox(label="Model URL", placeholder="HuggingFace/Pixeldrain zip link")
                model_name = gr.Textbox(label="Model Name", placeholder="Unique model name")
                download_btn = gr.Button("🌐 Download", variant="primary")
                dl_output_message = gr.Textbox()
                download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)
                gr.Examples(
                    examples=[
                        ["https://huggingface.co/phant0m4r/LiSA/resolve/main/LiSA.zip", "Lisa"],
                        ["https://pixeldrain.com/u/3tJmABXA", "Gura"],
                        ["https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models/resolve/main/AZKi%20(Hybrid).zip", "Azki"]
                    ],
                    inputs=[model_zip_link, model_name]
                )

            with gr.Tab("From Public Index"):
                gr.Markdown("### Instructions\nSelect a model to autofill URL and name, then download.")
                pub_zip_link = gr.Textbox(label="Model URL")
                pub_model_name = gr.Textbox(label="Model Name")
                download_pub_btn = gr.Button("🌐 Download", variant="primary")
                pub_dl_output_message = gr.Textbox()
                filter_tags = gr.CheckboxGroup(label="Filter by Tags")
                search_query = gr.Textbox(label="Search Models")
                load_public_models_button = gr.Button("Load Public Models", variant="primary")
                public_models_table = gr.DataFrame(headers=["Model Name", "Description", "Credit", "URL", "Tags"], interactive=False)

                load_public_models_button.click(
                    filter_available_public_models,
                    inputs=[gr.State(value=public_models), gr.State(value=voice_models)],
                    outputs=[public_models_table, filter_tags]
                )
                public_models_table.select(
                    lambda table, event: (table[event.index[0]][3], table[event.index[0]][0]),
                    inputs=[public_models_table],
                    outputs=[pub_zip_link, pub_model_name]
                )
                search_query.input(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                filter_tags.input(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                download_pub_btn.click(download_online_model, inputs=[pub_zip_link, pub_model_name], outputs=(pub_dl_output_message)

def create_upload_model_tab() -> None:
    """Create the Upload Model tab for the Gradio UI."""
    with gr.Tab("Upload Model"):
        gr.Markdown("### Upload RVC v2 Model\nZip file must contain .pth and optional .index files.")
        zip_file = gr.File(label="Zip File", file_types=[".zip"])
        local_model_name = gr.Textbox(label="Model Name", placeholder="Unique model name")
        model_upload_button = gr.Button("Upload Model", variant="primary")
        local_upload_output_message = gr.Markdown()
        model_upload_button.click(upload_local_model, inputs=[zip_file, local_model_name], outputs=[rvc_model, local_upload_output_message])

if __name__ == "__main__":
    parser = ArgumentParser(description="Generate an AI cover song.")
    parser.add_argument("--share", action="store_true", help="Enable sharing")
    parser.add_argument("--listen", action="store_true", help="Make WebUI reachable from local network.")
    parser.add_argument("--listen-host", type=str, help="Server hostname.")
    parser.add_argument("--listen-port", type=int, help="Server port.")
    args = parser.parse_args()

    voice_models = get_current_models(RVC_MODELS_DIR)
    public_models = load_public_models_data()

    with gr.Blocks(theme="Thatguy099/Sonix", title="AICoverGen WebUI") as app:
        gr.Label("# AICoverGen WebUI")
        visibility_state = gr.State(value=True)
        rvc_model, ai_cover = create_generate_tab(voice_models, visibility_state)
        create_download_model_tab(voice_models, public_models)
        create_upload_model_tab()

    app.launch(
        share=args.share,
        server_name=None if not args.listen else (args.listen_host or "0.0.0.0"),
        server_port=args.listen_port
    )
