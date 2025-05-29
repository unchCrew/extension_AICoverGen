import ffmpeg
import numpy as np
from huggingface_hub import hf_hub_download
import os
import requests
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')



def load_mdx():
    MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'

    if not os.path.exists(mdxnet_models_dir):
        os.makedirs(mdxnet_models_dir)
    
    mdx_model_names = ['UVR-MDX-NET-Voc_FT.onnx', 'UVR_MDXNET_KARA_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    
    for model in mdx_model_names:
        model_path = os.path.join(mdxnet_models_dir, model)
        if os.path.exists(model_path):
            print(f'{model} already exists, skipping download.')
            continue
            
        print(f'Starting download of {model}...')
        try:
            with requests.get(f'{MDX_DOWNLOAD_LINK}{model}', stream=True) as r:
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            print(f'Successfully downloaded {model}')
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
