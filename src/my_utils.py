import ffmpeg
import sox
import librosa
import numpy as np
from huggingface_hub import hf_hub_download
import os, gc, re
import gradio as gr
import requests
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment
import numpy as np
import hashlib
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

def raise_exception(error_msg, is_webui):
    if is_webui:
        raise gr.Error(error_msg)
    raise Exception(error_msg)


def display_progress(message, percent, is_webui, progress=None):
    if is_webui:
        progress(percent, desc=message)
    print("AICoverGen:" + message)



def load_mdx():
    MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'    
    mdx_model_names = ['UVR-MDX-NET-Voc_FT.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    
    for model in mdx_model_names:
        model_path = os.path.join(mdxnet_models_dir, model)
        if os.path.exists(model_path):
            pass
            continue
            
        
        try:
            with requests.get(f'{MDX_DOWNLOAD_LINK}{model}', stream=True) as r:
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            
        except requests.RequestException as e:
            print(f'Failed to download {model}: {e}')
    
    print('Model downloading process completed!')

def load_hubert_new(config, path=f"{rvc_models_dir}/hubert_base.pt"):
    from fairseq import checkpoint_utils

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config)
    hubert_model = hubert_model.float()
    return hubert_model.eval()


def get_and_load_hubert_new(config):
    hubert_path = hf_hub_download(
        repo_id="lj1995/VoiceConversionWebUI", filename="hubert_base.pt"
    )
    return load_hubert_new(config, hubert_path)


def download_rmvpe():
    local_dir = os.environ.get("rmvpe_root", rvc_models_dir)
    if not os.path.exists(os.path.join(local_dir, "rmvpe.pt")):
        print("Downloading rmvpe")
        file = hf_hub_download(
            repo_id="lj1995/VoiceConversionWebUI",
            filename="rmvpe.pt",
            local_dir=local_dir,
            local_dir_use_symlinks=False,
        )
        print(f"RMVPE downloaded to {os.environ.get('rmvpe_root')}")
        return file

def load_audio(file, sr):
    try:
        # https://github.com/openai/whisper/blob/main/whisper/audio.py#L26
        # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
        file = (
            file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        out, _ = (
            ffmpeg.input(file, threads=0)
            .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
            .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")

    return np.frombuffer(out, np.float32).flatten()



def add_audio_effects(audio_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping):
    output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'

    # Initialize audio effects plugins
    board = Pedalboard(
        [
            HighpassFilter(),
            Compressor(ratio=4, threshold_db=-15),
            Reverb(room_size=reverb_rm_size, dry_level=reverb_dry, wet_level=reverb_wet, damping=reverb_damping)
         ]
    )

    with AudioFile(audio_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
            # Read one second of audio at a time, until the file is empty:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

    return output_path


def convert_to_stereo(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)

    # check if mono
    if type(wave[0]) != np.ndarray:
        stereo_path = f'{os.path.splitext(audio_path)[0]}_stereo.wav'
        command = f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"'
        
        os.system(command)
        return stereo_path
    else:
        return audio_path


def pitch_shift(audio_path, pitch_change):
    output_path = f'{os.path.splitext(audio_path)[0]}_p{pitch_change}.wav'
    if not os.path.exists(output_path):
        y, sr = sf.read(audio_path)
        tfm = sox.Transformer()
        tfm.pitch(pitch_change)
        y_shifted = tfm.build_array(input_array=y, sample_rate_in=sr)
        sf.write(output_path, y_shifted, sr)

    return output_path


def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:11]

def combine_audio(audio_paths, output_path, main_gain, backup_gain, inst_gain, output_format):
    main_vocal_audio = AudioSegment.from_wav(audio_paths[0]) - 4 + main_gain
    backup_vocal_audio = AudioSegment.from_wav(audio_paths[1]) - 6 + backup_gain
    instrumental_audio = AudioSegment.from_wav(audio_paths[2]) - 7 + inst_gain
    main_vocal_audio.overlay(backup_vocal_audio).overlay(instrumental_audio).export(output_path, format=output_format)


