import sys
import webrtcvad
import numpy as np
import soundfile as sf
from scipy.signal import medfilt

def vad(s, fs, level):
    assert s.ndim == 1, "Only support mono pcm!"
    assert fs == 16000
    if s.dtype == np.float32 or s.dtype == np.float64:
        s *= 32768
        s = s.astype(np.int16)
    fralen = 30 * fs // 1000
    hop = fralen // 2
    franum = (s.shape[0] - fralen) // hop + 1
    frame_bytes = [s[(i*hop):(i*hop+fralen)].tostring() for i in range(franum)]  # string list:[franum, ]

    vad = webrtcvad.Vad(level)
    vad_results = np.array([vad.is_speech(frame, fs) for frame in frame_bytes])  # bool array:[franum, ]

    vad_out = np.repeat(vad_results, (hop, )).astype(np.float32) #
    vad_head = np.repeat(vad_out[0], (fralen-hop, ))
    vad_tail = np.repeat(vad_out[-1], (s.shape[0] - (franum-1)*hop-fralen, ))
    vad_out = np.concatenate((vad_head, vad_out, vad_tail), axis=0)

    return vad_out


if __name__ == "__main__":
    if len(sys.argv) == 2:
        filename = sys.argv[1]
    else:
        print(f"Usage: {sys.argv[0]} audio")
        exit(1)

    s, fs = sf.read(filename)
    vad_out = vad(s, fs, 3)
    sf.write("vad.wav", vad_out, fs, "PCM_16")