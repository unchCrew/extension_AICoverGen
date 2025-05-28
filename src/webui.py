import json
import os
import shutil
import urllib.request
import zipfile
from argparse import ArgumentParser

import gradio as gr
from main import song_cover_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'public_models.json', 'rmvpe.pt']
    return [item for item in models_list if item not in items_to_remove]

def update_models_list():
    return gr.Dropdown(choices=get_current_models(rvc_models_dir))

def load_public_models():
    models_table = [[model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])]
                    for model in public_models['voice_models'] if model['name'] not in voice_models]
    tags = list(public_models['tags'].keys())
    return models_table, tags

def extract_zip(extraction_folder, zip_name, progress=gr.Progress()):
    os.makedirs(extraction_folder, exist_ok=True)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, _, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)
            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
                model_filepath = os.path.join(root, name)

    if not model_filepath:
        raise gr.Error(f"No .pth model file found in {extraction_folder}.")
    
    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))
    
    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))

def download_online_model(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f"Downloading voice model: {dir_name}")
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f"Voice model directory {dir_name} already exists!")
        
        if 'pixeldrain.com' in url:
            url = f'https://pixeldrain.com/api/file/{zip_name}'
        
        urllib.request.urlretrieve(url, zip_name, reporthook=lambda b, bsize, tsize: progress(b * bsize / tsize))
        progress(0.5, desc="Extracting zip...")
        extract_zip(extraction_folder, zip_name)
        gr.Info(f"Model {dir_name} successfully downloaded!")
        return update_models_list()
    except Exception as e:
        raise gr.Error(str(e))

def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f"Voice model directory {dir_name} already exists!")
        
        progress(0.5, desc="Extracting zip...")
        extract_zip(extraction_folder, zip_path.name)
        gr.Info(f"Model {dir_name} successfully uploaded!")
        return update_models_list()
    except Exception as e:
        raise gr.Error(str(e))

def filter_models(tags, query):
    models_table = []
    for model in public_models['voice_models']:
        model_attributes = f"{model['name']} {model['description']} {model['credit']} {' '.join(model['tags'])}".lower()
        if (not tags or all(tag in model['tags'] for tag in tags)) and (not query or query.lower() in model_attributes):
            models_table.append([model['name'], model['description'], model['credit'], model['url'], ', '.join(model['tags'])])
    return models_table

def pub_dl_autofill(pub_models, event: gr.SelectData):
    return pub_models[event.index[0]][3], pub_models[event.index[0]][0]

def swap_visibility(state):
    return not state, state, "", None

def process_file_upload(file):
    return file.name, file.name

def show_hop_slider(pitch_detection_algo):
    return gr.Slider(visible=pitch_detection_algo == 'mangio-crepe')

if __name__ == '__main__':
    parser = ArgumentParser(description='Generate an AI cover song.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Enable sharing")
    parser.add_argument("--listen", action="store_true", default=False, help="Make WebUI reachable from local network.")
    parser.add_argument('--listen-host', type=str, help='Server hostname.')
    parser.add_argument('--listen-port', type=int, help='Server port.')
    args = parser.parse_args()

    voice_models = get_current_models(rvc_models_dir)
    with open(os.path.join(rvc_models_dir, 'public_models.json'), encoding='utf8') as infile:
        public_models = json.load(infile)

    # Apply modern Gradio theme
    theme = "Thatguy099/Sonix"

    with gr.Blocks(theme=theme, title="AICoverGen WebUI") as app:
        gr.Markdown("# AICoverGen WebUI", elem_classes=["header"])
        visibility_state = gr.State(value=True)  # Track visibility state

        with gr.Tabs():
            with gr.Tab("Generate"):
                with gr.Group(elem_classes=["section"]):
                    gr.Markdown("### Main Options")
                    with gr.Row(variant="compact"):
                        rvc_model = gr.Dropdown(voice_models, label="Voice Models", info="Add models to AICoverGen/rvc_models and refresh.", allow_custom_value=False)
                        ref_btn = gr.Button("🔄 Refresh Models", variant="primary")

                    with gr.Row(variant="compact"):
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

                    show_file_upload_button.click(swap_visibility, inputs=[visibility_state], outputs=[yt_link_col, file_upload_col, song_input, local_file], _js="() => {return {visibility_state: !visibility_state}}")
                    show_yt_link_button.click(swap_visibility, inputs=[visibility_state], outputs=[yt_link_col, file_upload_col, song_input, local_file], _js="() => {return {visibility_state: !visibility_state}}")
                    song_input_file.upload(process_file_upload, inputs=[song_input_file], outputs=[local_file, song_input])

                with gr.Accordion("Voice Conversion Options", open=False):
                    with gr.Row(variant="compact"):
                        index_rate = gr.Slider(0, 1, value=0.5, label="Index Rate", info="AI voice accent retention")
                        filter_radius = gr.Slider(0, 7, value=3, step=1, label="Filter Radius", info="Reduces breathiness if >=3")
                        rms_mix_rate = gr.Slider(0, 1, value=0.25, label="RMS Mix Rate", info="Original (0) vs. fixed loudness (1)")
                        protect = gr.Slider(0, 0.5, value=0.33, label="Protect Rate", info="Protects consonants/breath sounds")
                        f0_method = gr.Dropdown(['rmvpe', 'mangio-crepe'], value='rmvpe', label="Pitch Detection", info="rmvpe for clarity, mangio-crepe for smoothness")
                        crepe_hop_length = gr.Slider(32, 320, value=128, step=1, visible=False, label="Crepe Hop Length", info="Lower values improve pitch but risk cracks")
                    f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
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
                    output_format = gr.Dropdown(['mp3', 'wav'], value='mp3', label="Output Format", info="mp3: smaller, wav: higher quality")

                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    generate_btn = gr.Button("Generate", variant="primary")
                    ai_cover = gr.Audio(label="AI Cover", interactive=False)

                ref_btn.click(update_models_list, outputs=rvc_model)
                is_webui = gr.State(value=1)
                generate_btn.click(
                    song_cover_pipeline,
                    inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain, inst_gain,
                            index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length, protect, pitch_all,
                            reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, output_format],
                    outputs=[ai_cover]               
                )
                clear_btn.click(
                    lambda: [0, 0, 0, 0, 0.5, 3, 0.25, 0.33, 'rmvpe', 128, 0, 0.15, 0.2, 0.8, 0.7, 'mp3', None, ""],
                    outputs=[pitch, main_gain, backup_gain, inst_gain, index_rate, filter_radius, rms_mix_rate, protect,
                             f0_method, crepe_hop_length, pitch_all, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping,
                             output_format, ai_cover, song_input]
                )

            with gr.Tab("Download Model"):
                with gr.Tabs():
                    with gr.Tab("From URL"):
                        model_zip_link = gr.Textbox(label="Model URL", placeholder="HuggingFace/Pixeldrain zip link")
                        model_name = gr.Textbox(label="Model Name", placeholder="Unique model name")
                        download_btn = gr.Button("🌐 Download", variant="primary")
                        dl_output_message = gr.Markdown()
                        download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=[rvc_model, dl_output_message])
                        gr.Examples(
                            examples=[
                                ['https://huggingface.co/phant0m4r/LiSA/resolve/main/LiSA.zip', 'Lisa'],
                                ['https://pixeldrain.com/u/3tJmABXA', 'Gura'],
                                ['https://huggingface.co/Kit-Lemonfoot/kitlemonfoot_rvc_models/resolve/main/AZKi%20(Hybrid).zip', 'Azki']
                            ],
                            inputs=[model_zip_link, model_name]
                        )

                    with gr.Tab("From Public Index"):
                        gr.Markdown("### Instructions\nSelect a model to autofill URL and name, then download.")
                        pub_zip_link = gr.Textbox(label="Model URL")
                        pub_model_name = gr.Textbox(label="Model Name")
                        download_pub_btn = gr.Button("🌐 Download", variant="primary")
                        pub_dl_output_message = gr.Markdown()
                        filter_tags = gr.CheckboxGroup(label="Filter by Tags")
                        search_query = gr.Textbox(label="Search Models")
                        load_public_models_button = gr.Button("Load Public Models", variant="primary")
                        public_models_table = gr.DataFrame(headers=['Model Name', 'Description', 'Credit', 'URL', 'Tags'], interactive=False)
                        
                        load_public_models_button.click(load_public_models, outputs=[public_models_table, filter_tags])
                        public_models_table.select(pub_dl_autofill, inputs=[public_models_table], outputs=[pub_zip_link, pub_model_name])
                        search_query.input(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                        filter_tags.input(filter_models, inputs=[filter_tags, search_query], outputs=public_models_table)
                        download_pub_btn.click(download_online_model, inputs=[pub_zip_link, pub_model_name], outputs=[rvc_model, pub_dl_output_message])

            with gr.Tab("Upload Model"):
                gr.Markdown("### Upload RVC v2 Model\nZip file must contain .pth and optional .index files.")
                zip_file = gr.File(label="Zip File", file_types=[".zip"])
                local_model_name = gr.Textbox(label="Model Name", placeholder="Unique model name")
                model_upload_button = gr.Button("Upload Model", variant="primary")
                local_upload_output_message = gr.Markdown()
                model_upload_button.click(upload_local_model, inputs=[zip_file, local_model_name], outputs=[rvc_model, local_upload_output_message])

    app.launch(
        share=args.share_enabled,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port,
        queue=True
    )
