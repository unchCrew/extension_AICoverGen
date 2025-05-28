import ffmpeg
import numpy as np
from huggingface_hub import hf_hub_download
import os
from main import BASE_DIR
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')


def load_hubert_new(config, path=f"{rvc_models_dir}/hubert_base.pt"):
    from fairseq import checkpoint_utils

    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        [path],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
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
