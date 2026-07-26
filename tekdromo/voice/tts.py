# -*- coding: utf-8 -*-
"""
Voices. One interface, several implementations.

    v = load("piper")
    pcm16, rate = v.synth("hello")

Piper is what ships - it was measured at 0.72x real-time on this board, which
is faster than speech. The others exist because the interface costs nothing to
implement twice and a household assistant that cannot talk is useless: if the
model file is missing or onnxruntime fails to load, falling back to a native
synth is far better than falling back to silence.

Quality order, by ear: piper > pico > flite > espeak. espeak is kept anyway
because it is the phonemiser Piper depends on, so it is never an extra
dependency.
"""
import json
import os
import subprocess
import tempfile
import wave

import numpy as np

from . import pcm, phonemes

VOICE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "voices")
DEFAULT_MODEL = os.path.join(VOICE_DIR, "en_US-lessac-medium")


class Voice(object):
    """text -> (int16 samples, rate). Rate is the voice's own; callers resample
    only if they need to, and the speaker path deliberately does not."""

    name = "?"
    rate = pcm.RATE

    def synth(self, text):
        raise NotImplementedError

    def close(self):
        pass


# -- helpers ---------------------------------------------------------------
def _read_wav(path):
    w = wave.open(path, "rb")
    try:
        sr = w.getframerate()
        ch = w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    finally:
        w.close()
    if ch > 1:
        data = data.reshape(-1, ch).mean(axis=1).astype(np.int16)
    return data, sr


def _via_wav(cmd_builder):
    """Run a synth that can only write to a file, and read it back.

    pico2wave and flite have no stdout mode, so a temp file is unavoidable.
    Kept in one place so three implementations do not each grow their own
    version of the same tempfile dance.
    """
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        subprocess.check_call(cmd_builder(path), stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
        return _read_wav(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# -- implementations -------------------------------------------------------
class PiperVoice(Voice):
    """Neural (VITS) via onnxruntime, phonemised by espeak-ng.

    The ONNX session is built once and reused: loading the model costs ~4.6s
    and inference costs ~0.7x of the audio duration, so a per-utterance load
    would dominate everything and make short replies feel broken.
    """

    name = "piper"

    def __init__(self, model=DEFAULT_MODEL):
        import onnxruntime as ort
        self.model = model
        with open(model + ".onnx.json") as f:
            cfg = json.load(f)
        self.cfg = cfg
        self.rate = cfg["audio"]["sample_rate"]
        self.id_map = cfg["phoneme_id_map"]
        self.espeak = cfg.get("espeak", {}).get("voice", "en-us")
        inf = cfg.get("inference", {})
        self.scales = np.array([inf.get("noise_scale", 0.667),
                                inf.get("length_scale", 1.0),
                                inf.get("noise_w", 0.8)], dtype=np.float32)
        opts = ort.SessionOptions()
        # One thread per core we are willing to give it. The display needs the
        # rest, and oversubscribing makes the render loop stutter - which is
        # the one thing this project does not tolerate.
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 2
        self.sess = ort.InferenceSession(model + ".onnx", sess_options=opts,
                                         providers=["CPUExecutionProvider"])
        self.last_rounding = 0.0

    def synth(self, text):
        ph = phonemes.ipa(text, self.espeak)
        ids, missing = phonemes.to_ids(ph, self.id_map)
        self.last_rounding = phonemes.rounding(ph)
        if len(ids) <= 2:
            return pcm.silence(0), self.rate
        audio = self.sess.run(None, {
            "input": np.array([ids], dtype=np.int64),
            "input_lengths": np.array([len(ids)], dtype=np.int64),
            "scales": self.scales,
        })[0].squeeze()
        return pcm.from_float(audio), self.rate


class PicoVoice(Voice):
    """SVOX Pico. Notably more natural than espeak; 16 kHz."""

    name = "pico"

    def synth(self, text):
        return _via_wav(lambda p: ["pico2wave", "-l", "en-US", "-w", p, text])


class FliteVoice(Voice):
    name = "flite"

    def synth(self, text):
        return _via_wav(lambda p: ["flite", "-t", text, "-o", p])


class EspeakVoice(Voice):
    """Formant synthesis. Robotic, but instant and never fails."""

    name = "espeak"

    def __init__(self, wpm=150, voice="en-us"):
        self.wpm, self.voice = wpm, voice

    def synth(self, text):
        return _via_wav(lambda p: ["espeak-ng", "-v", self.voice,
                                   "-s", str(self.wpm), "-w", p, text])


VOICES = {"piper": PiperVoice, "pico": PicoVoice,
          "flite": FliteVoice, "espeak": EspeakVoice}
ORDER = ["piper", "pico", "flite", "espeak"]      # best first


def load(name=None, model=DEFAULT_MODEL):
    """Build a voice, falling back down the quality order if one won't start.

    Falling back is not hypothetical: Piper needs a 61 MB model file that is
    deliberately not in git, so a fresh checkout has no voice until it is
    fetched. Speaking badly beats not speaking.
    """
    names = [name] if name else ORDER
    errors = []
    for n in names:
        cls = VOICES.get(n)
        if cls is None:
            raise ValueError("unknown voice %r; have %s"
                             % (n, ", ".join(ORDER)))
        try:
            return cls(model) if n == "piper" else cls()
        except Exception as e:
            errors.append("%s: %s" % (n, e))
    raise RuntimeError("no voice available (%s)" % "; ".join(errors))


def available():
    """Which voices actually work here. Used by `tek voices`."""
    out = []
    for n in ORDER:
        try:
            v = VOICES[n](DEFAULT_MODEL) if n == "piper" else VOICES[n]()
            out.append((n, True, "%d Hz" % v.rate))
            v.close()
        except Exception as e:
            out.append((n, False, str(e)[:60]))
    return out
