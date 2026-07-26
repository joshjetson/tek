# -*- coding: utf-8 -*-
"""
Text -> phoneme IDs for Piper, without piper-phonemize.

piper-phonemize has no build for Python 3.6, which is what makes the official
Piper wrapper unusable on this box. It turns out not to matter: it is a C++
shim over espeak-ng, and espeak-ng is a native binary already in apt. So the
whole dependency reduces to running espeak-ng and looking the results up in the
phoneme_id_map that ships inside the voice's own .onnx.json.

Two things were measured rather than assumed:

  * espeak-ng 1.49.2 (2018, the version in Ubuntu 18.04) was expected to be too
    old and to emit IPA the model would not recognise. It gives 100% coverage
    against en_US-lessac-medium's map - 0 missing codepoints. No source build
    of 1.52 was needed.
  * espeak's --ipa output DROPS punctuation and emits a newline in its place.
    The model was trained WITH punctuation phonemes - they are its pause cues,
    and its map contains ',' '.' '?' '!' ';' ':'. Feeding it punctuation-free
    phonemes produces a flat, breathless monotone. So clauses are phonemised
    separately and the punctuation is put back between them.
"""
import re
import subprocess

# Reserved IDs, fixed by Piper's format: 0 pad, 1 start, 2 end, 3 word break.
PAD, BOS, EOS = 0, 1, 2

# Split on punctuation but KEEP it - re.split with a capturing group returns
# the separators, which is exactly the point here.
_CLAUSE = re.compile(r'([,.;:!?])')


def ipa(text, voice="en-us"):
    """Text -> an IPA phoneme string, punctuation preserved.

    Raises if espeak-ng is missing rather than returning empty: a silent
    fallback here would produce a voice that says nothing and looks like a
    broken audio path.
    """
    out = []
    parts = _CLAUSE.split(text)
    for i in range(0, len(parts), 2):
        chunk = parts[i].strip()
        punct = parts[i + 1] if i + 1 < len(parts) else ""
        if chunk:
            raw = subprocess.check_output(
                ["espeak-ng", "-q", "--ipa", "-v", voice, chunk],
                stderr=subprocess.DEVNULL).decode("utf-8")
            # espeak breaks clauses with newlines; collapse all whitespace to
            # single spaces, which the map has an entry for.
            out.append(" ".join(raw.split()))
        if punct:
            out.append(punct)
    return " ".join(out)


def to_ids(phoneme_str, id_map):
    """IPA string -> the model's phoneme IDs.

    Piper interleaves PAD between every phoneme and brackets the whole sequence
    with BOS/EOS. Unknown codepoints are skipped rather than mapped to PAD -
    inserting a pad would lengthen the utterance and slur the timing around the
    gap, whereas skipping only loses the sound itself.
    """
    ids = [BOS]
    missing = 0
    for ch in phoneme_str:
        got = id_map.get(ch)
        if got is None:
            missing += 1
            continue
        ids.extend(got)
        ids.append(PAD)
    ids.append(EOS)
    return ids, missing


def rounding(phoneme_str):
    """How rounded the lips are, 0..1, from the phoneme content.

    rig.speech.from_envelope() returns rounding=0 always, and says why: "real
    viseme shape needs phoneme information, which an envelope does not carry."
    The Piper path HAS that information - it had to be computed to synthesise
    at all - so the mouth can round on /u/, /o/, /w/ instead of only opening
    and closing. This is a per-utterance average, not a timeline; a timeline
    needs the model's duration predictor and is a later refinement.
    """
    if not phoneme_str:
        return 0.0
    rounded = u"uʊoɔɒwʉɵøyʏœɐɜ"
    hits = sum(1 for c in phoneme_str if c in rounded)
    vowels = sum(1 for c in phoneme_str if c in u"aeiouɪɛæɑʌəɚɜɔʊuːɐiy")
    if not vowels:
        return 0.0
    return min(1.0, float(hits) / vowels)
