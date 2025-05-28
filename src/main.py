import argparse
import gc
import hashlib
import json
import os
import shlex
import subprocess
from urllib.parse import urlparse, parse_qs
from contextlib import suppress
from pathlib import Path
from typing import Dict, Optional, Tuple
from huggingface_hub import hf_hub_download
import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import sox
import yt_dlp
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter
from pedalboard.io import AudioFile
from pydub import AudioSegment

from mdx import run_mdx
from rvc import Config, load_hubert, get_vc, rvc_infer

# Constants
BASE_DIR = Path(__file__).resolve().parent.parent
MDX_MODELS_DIR = BASE_DIR / 'mdxnet_models'
RVC_MODELS_DIR = BASE_DIR / 'rvc_models'
OUTPUT_DIR = BASE_DIR / 'song_output'
HUBERT_PATH = Path('')
COOKIES_PATH = Path('')

def get_hubert_path() -> Path:
    global HUBERT_PATH
    if not HUBERT_PATH:
        HUBERT_PATH = Path(hf_hub_download(
            repo_id="lj1995/VoiceConversionWebUI",
            filename="hubert_base.pt",
            local_dir=RVC_MODELS_DIR,
            local_dir_use_symlinks=False
        ))
    return HUBERT_PATH

def get_cookies_path() -> Optional[Path]:
    global COOKIES_PATH
    if not COOKIES_PATH:
        try:
            COOKIES_PATH = Path(hf_hub_download(
                repo_id="NeoPy/projects",
                filename="config.txt",
                local_dir=RVC_MODELS_DIR,
                local_dir_use_symlinks=False
            ))
            # Verify if the file is a valid Netscape cookies file
            if COOKIES_PATH.exists():
                with COOKIES_PATH.open('r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # Basic check for Netscape format (lines starting with domain or comments)
                    if not content or (not content.startswith('#') and '\t' not in content):
                        COOKIES_PATH = None
            else:
                COOKIES_PATH = None
        except Exception:
            COOKIES_PATH = None
    return COOKIES_PATH

def download_rmvpe() -> Path:
    rmvpe_path = RVC_MODELS_DIR / "rmvpe.pt"
    if not rmvpe_path.exists():
        print("Downloading rmvpe model...")
        rmvpe_path = Path(hf_hub_download(
            repo_id="lj1995/VoiceConversionWebUI",
            filename="rmvpe.pt",
            local_dir=RVC_MODELS_DIR,
            local_dir_use_symlinks=False
        ))
        print(f"RMVPE downloaded to {rmvpe_path}")
    return rmvpe_path

def get_youtube_video_id(url: str, ignore_playlist: bool = True) -> Optional[str]:
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        return query.path[1:] if query.path[1:] != 'watch' else parse_qs(query.query).get('v', [None])[0]
    
    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            return parse_qs(query.query).get('list', [None])[0]
        if query.path == '/watch':
            return parse_qs(query.query).get('v', [None])[0]
        if query.path.startswith(('/watch/', '/embed/', '/v/')):
            return query.path.split('/')[2]
    return None

def download_youtube(link: str, is_webui: bool) -> str:
    cookies_path = get_cookies_path()
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': str(OUTPUT_DIR / '%(title)s.%(ext)s'),  # Ensure output path is in OUTPUT_DIR
        'no_warnings': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
        'writesubtitles': False,
        'writeautomaticsub': False,
        'no_cookies': not cookies_path,  # Disable cookies if no valid cookies file
    }
    if cookies_path:
        ydl_opts['cookiefile'] = str(cookies_path)
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(link, download=True)
            output_path = ydl.prepare_filename(result)
            if not Path(output_path).exists():
                raise_error(f"Failed to download audio: {output_path} does not exist.", is_webui)
            return output_path
    except Exception as e:
        raise_error(f"YouTube download failed: {str(e)}. Ensure the URL is valid and the video is publicly accessible.", is_webui)

def raise_error(message: str, is_webui: bool) -> None:
    if

 is_webui:
        raise gr.Error(message)
    raise ValueError(message)

def get_rvc_model(voice_model: str, is_webui: bool) -> Tuple[str, str]:
    model_dir = RVC_MODELS_DIR / voice_model
    if not model_dir.exists():
        raise_error(f"RVC model folder {model_dir} does not exist.", is_webui)
    
    model_path = index_path = None
    for file in model_dir.iterdir():
        if file.suffix == '.pth':
            model_path = str(file)
        elif file.suffix == '.index':
            index_path = str(file)
    
    if not model_path:
        raise_error(f"No .pth model file found in {model_dir}", is_webui)
    return model_path, index_path or ''

def get_audio_paths(song_dir: Path) -> Dict[str, Optional[str]]:
    paths = {'orig': None, 'inst': None, 'main_vocals': None, 'backup_vocals': None}
    if not song_dir.exists():
        return paths
    for file in song_dir.iterdir():
        if file.name.endswith('_Instrumental.wav'):
            paths['inst'] = str(file)
            paths['orig'] = str(file).replace('_Instrumental', '')
        elif file.name.endswith('_Vocals_Main_DeReverb.wav'):
            paths['main_vocals'] = str(file)
        elif file.name.endswith('_Vocals_Backup.wav'):
            paths['backup_vocals'] = str(file)
    return paths

def to_stereo(audio_path: str) -> str:
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)
    if not isinstance(wave[0], np.ndarray):
        stereo_path = f"{Path(audio_path).stem}_stereo.wav"
        subprocess.run(shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"'), check=True)
        if not Path(stereo_path).exists():
            raise ValueError(f"Failed to convert to stereo: {stereo_path} does not exist.")
        return stereo_path
    return audio_path

def pitch_shift(audio_path: str, pitch_change: float) -> str:
    output_path = f"{Path(audio_path).stem}_p{pitch_change}.wav"
    if not Path(output_path).exists():
        y, sr = sf.read(audio_path)
        tfm = sox.Transformer()
        tfm.pitch(pitch_change)
        sf.write(output_path, tfm.build_array(input_array=y, sample_rate_in=sr), sr)
        if not Path(output_path).exists():
            raise ValueError(f"Failed to apply pitch shift: {output_path} does not exist.")
    return output_path

def get_file_hash(filepath: str) -> str:
    hasher = hashlib.blake2b()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()[:11]

def show_progress(message: str, percent: float, is_webui: bool, progress: Optional[gr.Progress] = None) -> None:
    if is_webui and progress:
        try:
            progress(percent, desc=message)
        except (IndexError, AttributeError):
            print(f"[WebUI Progress Error] {message} ({percent*100:.0f}%)")
    else:
        print(f"{message} ({percent*100:.0f}%)")

def preprocess_song(song_input: str, mdx_model_params: dict, song_id: str, is_webui: bool, input_type: str, progress: Optional[gr.Progress] = None) -> Dict[str, str]:
    song_dir = OUTPUT_DIR / song_id
    song_dir.mkdir(parents=True, exist_ok=True)
    
    show_progress('Preparing audio input...', 0, is_webui, progress)
    orig_song_path = download_youtube(song_input.split('&')[0], is_webui) if input_type == 'yt' else song_input
    if not Path(orig_song_path).exists():
        raise_error(f"Audio file {orig_song_path} not found after download.", is_webui)
    orig_song_path = to_stereo(orig_song_path)

    show_progress('Separating vocals from instrumental...', 0.2, is_webui, progress)
    vocals_path, inst_path = run_mdx(
        mdx_model_params, str(song_dir), str(MDX_MODELS_DIR / 'UVR-MDX-NET-Voc_FT.onnx'),
        orig_song_path, denoise=True, keep_orig=input_type == 'local'
    )
    if not (Path(vocals_path).exists() and Path(inst_path).exists()):
        raise_error("Failed to separate vocals and instrumental.", is_webui)

    show_progress('Separating main and backup vocals...', 0.4, is_webui, progress)
    backup_vocals, main_vocals = run_mdx(
        mdx_model_params, str(song_dir), str(MDX_MODELS_DIR / 'UVR_MDXNET_KARA_2.onnx'),
        vocals_path, suffix='Backup', invert_suffix='Main', denoise=True
    )
    if not (Path(backup_vocals).exists() and Path(main_vocals).exists()):
        raise_error("Failed to separate main and backup vocals.", is_webui)

    show_progress('Applying dereverb to main vocals...', 0.6, is_webui, progress)
    _, main_vocals_dereverb = run_mdx(
        mdx_model_params, str(song_dir), str(MDX_MODELS_DIR / 'Reverb_HQ_By_FoxJoy.onnx'),
        main_vocals, invert_suffix='DeReverb', exclude_main=True, denoise=True
    )
    if not Path(main_vocals_dereverb).exists():
        raise_error("Failed to apply dereverb to main vocals.", is_webui)

    return {
        'orig': orig_song_path,
        'vocals': vocals_path,
        'inst': inst_path,
        'main_vocals': main_vocals,
        'backup_vocals': backup_vocals,
        'main_vocals_dereverb': main_vocals_dereverb
    }

def voice_change(voice_model: str, vocals_path: str, output_path: str, pitch_change: float, f0_method: str, index_rate: float, 
                 filter_radius: int, rms_mix_rate: float, protect: float, crepe_hop_length: int, is_webui: bool) -> None:
    show_progress('Converting voice using RVC...', 0.7, is_webui)
    model_path, index_path = get_rvc_model(voice_model, is_webui)
    config = Config(device='cuda:0', is_half=True)
    
    try:
        hubert_model = load_hubert(config.device, config.is_half, str(get_hubert_path()))
        cpt, version, net_g, tgt_sr, vc = get_vc(config.device, config.is_half, config, model_path)
        rvc_infer(index_path, index_rate, vocals_path, output_path, pitch_change, f0_method,
                  cpt, version, net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model)
        if not Path(output_path).exists():
            raise_error(f"Voice conversion failed: {output_path} does not exist.", is_webui)
    finally:
        del hubert_model, cpt, vc, net_g
        gc.collect()

def add_audio_effects(audio_path: str, reverb_rm_size: float, reverb_wet: float, reverb_dry: float, reverb_damping: float) -> str:
    output_path = f"{Path(audio_path).stem}_mixed.wav"
    board = Pedalboard([
        HighpassFilter(),
        Compressor(ratio=4, threshold_db=-15),
        Reverb(room_size=reverb_rm_size, dry_level=reverb_dry, wet_level=reverb_wet, damping=reverb_damping)
    ])
    
    with AudioFile(audio_path) as f, AudioFile(output_path, 'w', f.samplerate, f.num_channels) as o:
        while f.tell() < f.frames:
            chunk = f.read(int(f.samplerate))
            o.write(board(chunk, f.samplerate, reset=False))
    if not Path(output_path).exists():
        raise ValueError(f"Failed to apply audio effects: {output_path} does not exist.")
    return output_path

def combine_audio(audio_paths: list, output_path: str, main_gain: float, backup_gain: float, inst_gain: float, output_format: str) -> None:
    main_vocal = AudioSegment.from_wav(audio_paths[0]) - 4 + main_gain
    backup_vocal = AudioSegment.from_wav(audio_paths[1]) - 6 + backup_gain
    instrumental = AudioSegment.from_wav(audio_paths[2]) - 7 + inst_gain
    main_vocal.overlay(backup_vocal).overlay(instrumental).export(output_path, format=output_format)
    if not Path(output_path).exists():
        raise ValueError(f"Failed to combine audio: {output_path} does not exist.")

def song_cover_pipeline(song_input: str, voice_model: str, pitch_change: float, keep_files: bool, is_webui: bool = False, 
                       main_gain: float = 0, backup_gain: float = 0, inst_gain: float = 0, index_rate: float = 0.5, 
                       filter_radius: int = 3, rms_mix_rate: float = 0.25, f0_method: str = 'rmvpe', 
                       crepe_hop_length: int = 128, protect: float = 0.33, pitch_change_all: float = 0,
                       reverb_rm_size: float = 0.15, reverb_wet: float = 0.2, reverb_dry: float = 0.8, 
                       reverb_damping: float = 0.7, output_format: str = 'mp3', progress: Optional[gr.Progress] = None) -> str:
    try:
        if not song_input or not voice_model:
            raise_error("Song input and voice model are required.", is_webui)
        
        show_progress('Starting AI cover generation...', 0, is_webui, progress)
        
        with (MDX_MODELS_DIR / 'model_data.json').open() as f:
            mdx_model_params = json.load(f)
        
        download_rmvpe()
        input_type = 'yt' if urlparse(song_input).scheme in ('http', 'https') else 'local'
        
        if input_type == 'yt':
            song_id = get_youtube_video_id(song_input)
            if not song_id:
                raise_error("Invalid YouTube URL.", is_webui)
        else:
            song_input = song_input.strip('"')
            if not Path(song_input).exists():
                raise_error(f"File {song_input} does not exist.", is_webui)
            song_id = get_file_hash(song_input)
        
        song_dir = OUTPUT_DIR / song_id
        song_dir.mkdir(parents=True, exist_ok=True)
        audio_paths = get_audio_paths(song_dir)
        
        if not audio_paths['orig'] or not keep_files:
            audio_paths = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress)
        
        pitch_change = pitch_change * 12 + pitch_change_all
        ai_vocals_path = song_dir / f"{Path(audio_paths['orig']).stem}_{voice_model}_p{pitch_change}_i{index_rate}_fr{filter_radius}_rms{rms_mix_rate}_pro{protect}_{f0_method}{'' if f0_method != 'mangio-crepe' else f'_{crepe_hop_length}'}.wav"
        ai_cover_path = song_dir / f"{Path(audio_paths['orig']).stem} ({voice_model} Ver).{output_format}"
        
        if not ai_vocals_path.exists():
            voice_change(voice_model, audio_paths['main_vocals_dereverb'], str(ai_vocals_path), pitch_change,
                        f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui)
        
        show_progress('Applying audio effects to vocals...', 0.8, is_webui, progress)
        ai_vocals_mixed = add_audio_effects(str(ai_vocals_path), reverb_rm_size, reverb_wet, reverb_dry, reverb_damping)
        
        inst_path = audio_paths['inst']
        backup_vocals_path = audio_paths['backup_vocals']
        if pitch_change_all != 0:
            show_progress('Applying overall pitch change...', 0.85, is_webui, progress)
            inst_path = pitch_shift(inst_path, pitch_change_all)
            backup_vocals_path = pitch_shift(backup_vocals_path, pitch_change_all)
        
        show_progress('Combining audio tracks...', 0.9, is_webui, progress)
        combine_audio([ai_vocals_mixed, backup_vocals_path, inst_path], str(ai_cover_path),
                     main_gain, backup_gain, inst_gain, output_format)
        
        if not keep_files:
            show_progress('Cleaning up intermediate files...', 0.95, is_webui, progress)
            for file in [audio_paths['vocals'], audio_paths['main_vocals'], ai_vocals_mixed] + \
                       ([inst_path, backup_vocals_path] if pitch_change_all != 0 else []):
                if file and Path(file).exists():
                    Path(file).unlink()
        
        show_progress('AI cover generated successfully!', 1.0, is_webui, progress)
        return str(ai_cover_path)
    
    except Exception as e:
        raise_error(str(e), is_webui)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate an AI cover song.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--song-input', type=str, required=True, help='YouTube URL or path to local mp3/wav file')
    parser.add_argument('-dir', '--rvc-dirname', type=str, required=True, help='Folder name in rvc_models containing the RVC model')
    parser.add_argument('-p', '--pitch-change', type=int, required=True, help='Pitch change for AI vocals (octaves, e.g., 1 for male to female)')
    parser.add_argument('-k', '--keep-files', action='store_true', help='Keep intermediate audio files')
    parser.add_argument('-ir', '--index-rate', type=float, default=0.5, help='Timbre retention (0 to 1)')
    parser.add_argument('-fr', '--filter-radius', type=int, default=3, help='Median filter radius (0 to 7) for pitch results')
    parser.add_argument('-rms', '--rms-mix-rate', type=float, default=0.25, help='Original vs fixed loudness (0 to 1)')
    parser.add_argument('-palgo', '--pitch-detection-algo', type=str, default='rmvpe', choices=['rmvpe', 'mangio-crepe'], help='Pitch detection algorithm')
    parser.add_argument('-hop', '--crepe-hop-length', type=int, default=128, help='Pitch check frequency for mangio-crepe (ms)')
    parser.add_argument('-pro', '--protect', type=float, default=0.33, help='Protect voiceless consonants (0 to 0.5)')
    parser.add_argument('-mv', '--main-vol', type=int, default=0, help='Main vocals volume change (dB)')
    parser.add_argument('-bv', '--backup-vol', type=int, default=0, help='Backup vocals volume change (dB)')
    parser.add_argument('-iv', '--inst-vol', type=int, default=0, help='Instrumental volume change (dB)')
    parser.add_argument('-pall', '--pitch-change-all', type=int, default=0, help='Pitch change for all tracks (octaves)')
    parser.add_argument('-rsize', '--reverb-size', type=float, default=0.15, help='Reverb room size (0 to 1)')
    parser.add_argument('-rwet', '--reverb-wetness', type=float, default=0.2, help='Reverb wet level (0 to 1)')
    parser.add_argument('-rdry', '--reverb-dryness', type=float, default=0.8, help='Reverb dry level (0 to 1)')
    parser.add_argument('-rdamp', '--reverb-damping', type=float, default=0.7, help='Reverb damping (0 to 1)')
    parser.add_argument('-oformat', '--output-format', type=str, default='mp3', choices=['mp3', 'wav'], help='Output audio format')
    
    args = parser.parse_args()
    
    cover_path = song_cover_pipeline(
        args.song_input, args.rvc_dirname, args.pitch_change, args.keep_files,
        main_gain=args.main_vol, backup_gain=args.backup_vol, inst_gain=args.inst_vol,
        index_rate=args.index_rate, filter_radius=args.filter_radius, rms_mix_rate=args.rms_mix_rate,
        f0_method=args.pitch_detection_algo, crepe_hop_length=args.crepe_hop_length, protect=args.protect,
        pitch_change_all=args.pitch_change_all, reverb_rm_size=args.reverb_size,
        reverb_wet=args.reverb_wetness, reverb_dry=args.reverb_dryness, reverb_damping=args.reverb_damping,
        output_format=args.output_format
    )
    print(f"[+] Cover generated at {cover_path}")
