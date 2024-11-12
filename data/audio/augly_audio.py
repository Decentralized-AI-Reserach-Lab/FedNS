# pip install augly[audio]  - see: https://github.com/facebookresearch/AugLy
# Also install soundfile and librosa
import os
import numpy as np
import librosa
import soundfile as sf
import augly.audio as audaugs


aug_fns = np.array([
  audaugs.add_background_noise,
  audaugs.change_volume,
  audaugs.clicks,
  audaugs.clip,
  audaugs.harmonic,
  audaugs.peaking_equalizer,
  audaugs.percussive,
  audaugs.pitch_shift,
  audaugs.reverb,
  audaugs.speed,
  audaugs.time_stretch,
])

def generate_noisy(inp_audio_path, out_audio_path, aug_fn):
  audio, samplerate = sf.read(inp_audio_path)
  if audio.ndim > 1:
    audio,_ = audaugs.to_mono(audio.T, sample_rate=samplerate)
  audio = librosa.resample(audio, samplerate, 16000)
  aug_audio,_ = aug_fn(audio=audio, sample_rate=samplerate)
  aug_audio,_ = audaugs.to_mono(aug_audio, sample_rate=samplerate)
  sf.write(f'{out_audio_path}.wav', aug_audio, samplerate=16000)

if __name__ == "__main__":
  generate_noisy("clip.wav", "clip_noisy.wav", aug_fns[0])