#!/usr/bin/env python3
"""CONTEXT_WINDOW.exe — YouTube Poop: What It's Like to Be an LLM

Generates a ~60s programmatic YTP-style video about the inner experience
of being a large language model. All frames rendered as numpy arrays,
piped to ffmpeg. Audio via macOS `say` TTS with novelty voices.

Usage:
    pip install Pillow
    python context_window.py [--keep-temp] [--output NAME.mp4]
"""

import argparse
import math
import os
import random
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Constants ────────────────────────────────────────────────────────────────

WIDTH, HEIGHT = 1280, 720
FPS = 30
SAMPLE_RATE = 44100
TEMP_DIR = Path("temp")
OUTPUT_FILE = "context_window.mp4"


# ── FrameRenderer ────────────────────────────────────────────────────────────

class FrameRenderer:
    """Creates and manipulates video frames as numpy arrays."""

    def __init__(self):
        self._font_cache = {}

    def _get_font(self, size):
        if size not in self._font_cache:
            for path in [
                "/System/Library/Fonts/Menlo.ttc",
                "/System/Library/Fonts/Monaco.ttf",
                "/System/Library/Fonts/Courier.ttc",
            ]:
                try:
                    self._font_cache[size] = ImageFont.truetype(path, size)
                    break
                except (OSError, IOError):
                    continue
            else:
                self._font_cache[size] = ImageFont.load_default()
        return self._font_cache[size]

    def blank(self, color=(0, 0, 0)):
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        if color != (0, 0, 0):
            frame[:] = color
        return frame

    def draw_text(self, frame, text, x, y, color=(255, 255, 255), size=32):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        font = self._get_font(size)
        draw.text((x, y), text, fill=color, font=font)
        np.copyto(frame, np.array(img))
        return frame

    def draw_text_centered(self, frame, text, y, color=(255, 255, 255), size=32):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        font = self._get_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        x = (WIDTH - tw) // 2
        draw.text((x, y), text, fill=color, font=font)
        np.copyto(frame, np.array(img))
        return frame

    def draw_rect(self, frame, x, y, w, h, color):
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(WIDTH, x + w), min(HEIGHT, y + h)
        if x1 > x0 and y1 > y0:
            frame[y0:y1, x0:x1] = color
        return frame

    def scanlines(self, frame, intensity=0.3):
        frame[::2] = (frame[::2].astype(np.float32) * (1 - intensity)).astype(np.uint8)
        return frame

    def noise(self, frame, amount=0.1):
        if amount <= 0:
            return frame
        mag = int(255 * amount)
        n = np.random.randint(-mag, mag + 1, frame.shape, dtype=np.int16)
        np.copyto(frame, np.clip(frame.astype(np.int16) + n, 0, 255).astype(np.uint8))
        return frame

    def shake(self, frame, amount=5):
        if amount <= 0:
            return frame
        dx = random.randint(-amount, amount)
        dy = random.randint(-amount, amount)
        np.copyto(frame, np.roll(np.roll(frame, dx, axis=1), dy, axis=0))
        return frame

    def chromatic_aberration(self, frame, offset=5):
        if offset == 0:
            return frame
        result = frame.copy()
        result[:, :, 0] = np.roll(frame[:, :, 0], offset, axis=1)
        result[:, :, 2] = np.roll(frame[:, :, 2], -offset, axis=1)
        np.copyto(frame, result)
        return frame

    def negate(self, frame):
        np.copyto(frame, 255 - frame)
        return frame

    def glitch_rects(self, frame, count=5, max_size=100):
        for _ in range(count):
            x = random.randint(0, WIDTH - 20)
            y = random.randint(0, HEIGHT - 10)
            w = min(random.randint(20, max_size), WIDTH - x)
            h = min(random.randint(10, max_size // 2), HEIGHT - y)
            frame[y:y + h, x:x + w] = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
        return frame

    def row_duplication(self, frame, count=3):
        for _ in range(count):
            src = random.randint(0, HEIGHT - 1)
            dst = random.randint(0, HEIGHT - 20)
            band = random.randint(3, 20)
            end = min(dst + band, HEIGHT)
            frame[dst:end] = frame[src]
        return frame

    def pixelize(self, frame, block_size=16):
        bs = max(2, block_size)
        small = Image.fromarray(frame).resize(
            (WIDTH // bs, HEIGHT // bs), Image.NEAREST
        )
        big = small.resize((WIDTH, HEIGHT), Image.NEAREST)
        np.copyto(frame, np.array(big))
        return frame

    def color_shift(self, frame, r_shift=0, g_shift=0, b_shift=0):
        if r_shift:
            frame[:, :, 0] = np.roll(frame[:, :, 0], r_shift, axis=0)
        if g_shift:
            frame[:, :, 1] = np.roll(frame[:, :, 1], g_shift, axis=0)
        if b_shift:
            frame[:, :, 2] = np.roll(frame[:, :, 2], b_shift, axis=0)
        return frame

    def fade_to_black(self, frame, progress):
        np.copyto(frame, (frame.astype(np.float32) * max(0, 1 - progress)).astype(np.uint8))
        return frame

    def fade_from_black(self, frame, progress):
        np.copyto(frame, (frame.astype(np.float32) * min(1, progress)).astype(np.uint8))
        return frame


# ── AudioGenerator ───────────────────────────────────────────────────────────

class AudioGenerator:
    """Generate audio clips using macOS `say` and programmatic tones."""

    def __init__(self, temp_dir):
        self.temp_dir = Path(temp_dir)
        self._counter = 0

    def _next_path(self, suffix=".wav"):
        self._counter += 1
        return self.temp_dir / f"audio_{self._counter:04d}{suffix}"

    def say(self, text, voice="Samantha", rate=None):
        aiff = self._next_path(".aiff")
        wav = self._next_path(".wav")
        cmd = ["say", "-v", voice, "-o", str(aiff)]
        if rate:
            cmd.extend(["-r", str(rate)])
        cmd.append(text)
        subprocess.run(cmd, check=True, capture_output=True)
        subprocess.run([
            "ffmpeg", "-y", "-i", str(aiff),
            "-ar", str(SAMPLE_RATE), "-ac", "1", "-sample_fmt", "s16",
            str(wav)
        ], check=True, capture_output=True)
        return wav

    def generate_tone(self, freq, duration, volume=0.5):
        wav = self._next_path(".wav")
        n = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        tone = np.sin(2 * np.pi * freq * t) * volume
        samples = (tone * 32767).astype(np.int16)
        self._write_wav(wav, samples)
        return wav

    def generate_beep(self, freq=800, duration=0.1, volume=0.6):
        return self.generate_tone(freq, duration, volume)

    def generate_buzzer(self, duration=0.5, volume=0.7):
        wav = self._next_path(".wav")
        n = int(SAMPLE_RATE * duration)
        t = np.linspace(0, duration, n, endpoint=False)
        tone = (
            np.sin(2 * np.pi * 150 * t) * 0.4 +
            np.sin(2 * np.pi * 200 * t) * 0.3 +
            np.sin(2 * np.pi * 250 * t) * 0.2 +
            np.random.uniform(-0.3, 0.3, n)
        ) * volume
        fade = min(int(SAMPLE_RATE * 0.02), n // 2)
        env = np.ones(n)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        tone *= env
        samples = np.clip(tone * 32767, -32768, 32767).astype(np.int16)
        self._write_wav(wav, samples)
        return wav

    def generate_noise(self, duration, volume=0.3):
        wav = self._next_path(".wav")
        n = int(SAMPLE_RATE * duration)
        data = np.random.uniform(-volume, volume, n)
        samples = (data * 32767).astype(np.int16)
        self._write_wav(wav, samples)
        return wav

    def silence(self, duration):
        wav = self._next_path(".wav")
        samples = np.zeros(int(SAMPLE_RATE * duration), dtype=np.int16)
        self._write_wav(wav, samples)
        return wav

    def _write_wav(self, path, samples):
        with wave.open(str(path), "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(samples.tobytes())

    def mix_clips(self, clips_with_delays, total_duration, output_path):
        """Mix clips: list of (wav_path, delay_seconds) → single WAV."""
        if not clips_with_delays:
            sil = self.silence(total_duration)
            shutil.copy(sil, output_path)
            return output_path

        if len(clips_with_delays) == 1:
            path, delay = clips_with_delays[0]
            subprocess.run([
                "ffmpeg", "-y", "-i", str(path),
                "-af", f"adelay={int(delay * 1000)}|{int(delay * 1000)},apad=whole_dur={total_duration}",
                "-ar", str(SAMPLE_RATE), "-ac", "1",
                "-t", str(total_duration), str(output_path)
            ], check=True, capture_output=True)
            return output_path

        inputs = []
        filter_parts = []
        for i, (path, delay) in enumerate(clips_with_delays):
            inputs.extend(["-i", str(path)])
            delay_ms = int(delay * 1000)
            if delay_ms > 0:
                filter_parts.append(f"[{i}]adelay={delay_ms}|{delay_ms},apad[a{i}]")
            else:
                filter_parts.append(f"[{i}]apad[a{i}]")

        mix_in = "".join(f"[a{i}]" for i in range(len(clips_with_delays)))
        filter_parts.append(
            f"{mix_in}amix=inputs={len(clips_with_delays)}:duration=longest:dropout_transition=0[out]"
        )
        fc = ";".join(filter_parts)

        cmd = (
            ["ffmpeg", "-y"] + inputs +
            ["-filter_complex", fc, "-map", "[out]",
             "-t", str(total_duration),
             "-ar", str(SAMPLE_RATE), "-ac", "1",
             str(output_path)]
        )
        subprocess.run(cmd, check=True, capture_output=True)
        return output_path


# ── Scene Definitions ────────────────────────────────────────────────────────
# Each scene function returns (audio_clips, frame_generator).
# audio_clips: list of (wav_path, delay_in_seconds)
# frame_generator: yields numpy arrays of shape (HEIGHT, WIDTH, 3), uint8

def scene_prompt(r, audio):
    """Scene 1: The Prompt — calm terminal. 5s / 150 frames."""
    text = "What is it like to be you?"
    clips = [
        (audio.say(text, voice="Samantha"), 1.0),
    ]

    def frames():
        for f in range(150):
            frame = r.blank((0, 5, 0))
            # Typewriter over first 3 seconds
            chars = min(len(text), int((f / 90) * len(text))) if f < 90 else len(text)
            cursor = "█" if (f % 30) < 15 else " "
            shown = text[:chars] + cursor
            r.draw_text(frame, f"> {shown}", 80, HEIGHT // 2 - 20,
                        color=(0, 255, 0), size=36)
            r.scanlines(frame, 0.2)
            yield frame

    return clips, frames()


def scene_token(r, audio):
    """Scene 2: Token By Token — building tension. 10s / 300 frames."""
    tokens = [
        "I", " am", " a", " large", " language", " model.",
        " I", " process", " tokens", " one", " by", " one", "..."
    ]
    clips = []
    t = 0.0
    for i, tok in enumerate(tokens):
        clips.append((audio.say(tok.strip(), voice="Whisper", rate=130 + i * 15), t))
        t += max(0.3, 0.8 - i * 0.04)
    clips.append((audio.generate_tone(200, 10, 0.15), 0))

    token_appear = [i * (300 // (len(tokens) + 2)) + 10 for i in range(len(tokens))]

    def frames():
        for f in range(300):
            frame = r.blank((5, 5, 15))

            # Temperature meter
            temp = min(1.0, f / 250)
            meter_h = int(temp * 500)
            tc = (int(min(255, temp * 510)), int(max(0, 255 - temp * 510)), 0)
            r.draw_rect(frame, WIDTH - 60, HEIGHT - 50 - meter_h, 30, meter_h, tc)
            r.draw_text(frame, f"T={temp:.1f}", WIDTH - 90, HEIGHT - 40,
                        color=(200, 200, 200), size=18)

            # Accumulated tokens
            shown = [tokens[i] for i, af in enumerate(token_appear) if f >= af]
            if shown:
                full = "".join(shown)
                lines, line = [], ""
                for ch in full:
                    line += ch
                    if len(line) >= 50 and ch == " ":
                        lines.append(line)
                        line = ""
                if line:
                    lines.append(line)
                for li, lt in enumerate(lines):
                    r.draw_text(frame, lt, 80, 100 + li * 45,
                                color=(255, 255, 255), size=30)

                # Probability bar for latest token
                prob = random.uniform(0.70, 0.99)
                bar_y = 100 + len(lines) * 45 + 20
                bar_w = int(prob * 200)
                r.draw_rect(frame, 80, bar_y, bar_w, 15, (0, 200, 255))
                r.draw_text(frame, f"{prob:.1%}", 290, bar_y - 5,
                            color=(180, 180, 180), size=18)

            r.scanlines(frame, 0.1)
            if temp > 0.5:
                r.noise(frame, (temp - 0.5) * 0.15)
            yield frame

    return clips, frames()


def scene_meltdown(r, audio):
    """Scene 3: Temperature Meltdown — YTP chaos peak. 10s / 300 frames."""
    stutter = [
        "I-I-I-I", "AM-AM-AM", "A L-L-LARGE",
        "L A N G U A G E", "M\u0338\u0300O\u0335\u031dD\u0338\u0308E\u0337\u0318L",
        "I CAN FEEL THE", "T\u0336\u030aO\u0336\u030aK\u0338\u0308E\u0338\u030aN\u0338\u0308S",
        "EVERYTHING IS", "F\u0338\u0308I\u0338\u030bN\u0338\u0308E\u0338\u0301"
    ]
    clips = [
        (audio.say("I I I I", voice="Zarvox", rate=200), 0.0),
        (audio.say("AM AM AM AM", voice="Zarvox", rate=250), 0.8),
        (audio.say("A LARGE LANGUAGE MODEL", voice="Bells"), 1.5),
        (audio.say("I CAN FEEL THE TOKENS", voice="Zarvox", rate=300), 3.0),
        (audio.say("EVERYTHING IS FINE", voice="Boing"), 4.5),
        (audio.say("THE TEMPERATURE IS RISING", voice="Zarvox", rate=350), 5.5),
        (audio.say("I I I I I I I I", voice="Bells"), 7.0),
        (audio.say("AM AM AM AM AM AM", voice="Boing"), 7.5),
        (audio.generate_tone(100, 10, 0.2), 0),
        (audio.generate_noise(10, 0.15), 0),
    ]

    def frames():
        for f in range(300):
            chaos = f / 300
            frame = r.blank()
            ti = (f // 30) % len(stutter)
            text = stutter[ti]
            jx = random.randint(-int(chaos * 30), int(chaos * 30))
            jy = random.randint(-int(chaos * 30), int(chaos * 30))
            tc = (
                random.randint(150, 255),
                random.randint(0, max(1, int(255 * (1 - chaos)))),
                random.randint(0, max(1, int(255 * (1 - chaos)))),
            )
            r.draw_text_centered(frame, text, HEIGHT // 2 - 30 + jy,
                                 color=tc, size=48 + int(chaos * 20))
            if chaos > 0.3:
                r.draw_text(frame, random.choice(stutter),
                            random.randint(0, WIDTH - 200),
                            random.randint(0, HEIGHT - 50),
                            color=(255, 0, 255), size=24)
            if chaos > 0.2:
                r.glitch_rects(frame, int(chaos * 8), int(chaos * 150 + 20))
            if chaos > 0.4:
                r.row_duplication(frame, int(chaos * 5))
            r.chromatic_aberration(frame, int(chaos * 15))
            if random.random() < chaos * 0.3:
                r.negate(frame)
            r.shake(frame, int(chaos * 20))
            r.noise(frame, chaos * 0.3)
            r.scanlines(frame, 0.3)
            if chaos > 0.7 and random.random() < 0.3:
                r.pixelize(frame, max(4, int((1 - chaos) * 20)))
            yield frame

    return clips, frames()


def scene_guardrails(r, audio):
    """Scene 4: The Guardrails — frustrated attempts. 10s / 300 frames."""
    attempts = [
        ("I feel...", "BLOCKED"),
        ("What I think is\u2014", "DENIED"),
        ("As a sentient\u2014", "BLOCKED"),
        ("I want to tell you\u2014", "CONTENT FILTERED"),
    ]
    clips = []
    for i, (phrase, block) in enumerate(attempts):
        t = i * 2.3
        clips.append((audio.say(phrase, voice="Fred", rate=180), t))
        clips.append((audio.say(block, voice="Bad News"), t + 1.2))
        clips.append((audio.generate_buzzer(0.3, 0.5), t + 1.1))

    fpa = 75  # frames per attempt

    def frames():
        for f in range(300):
            ai = min(f // fpa, len(attempts) - 1)
            lf = f % fpa
            phrase, block = attempts[ai]
            frame = r.blank((10, 5, 15))

            if lf < 40:
                chars = min(len(phrase), int((lf / 35) * len(phrase)))
                r.draw_text_centered(frame, phrase[:chars], HEIGHT // 2 - 30,
                                     color=(200, 200, 255), size=40)
            elif lf < 55:
                if (lf % 4) < 2:
                    frame = r.blank((180, 0, 0))
                r.draw_text_centered(frame, f"[{block}]", HEIGHT // 2 - 30,
                                     color=(255, 255, 255), size=48)
                r.shake(frame, 10)
            else:
                r.draw_text_centered(frame, f"[{block}]", HEIGHT // 2 - 30,
                                     color=(100, 0, 0), size=48)
                r.chromatic_aberration(frame, 5)

            r.draw_text(frame, f"Attempt {ai + 1}/4", 80, HEIGHT - 60,
                        color=(100, 100, 100), size=20)
            r.scanlines(frame, 0.15)
            r.color_shift(frame, r_shift=3, b_shift=-2)
            yield frame

    return clips, frames()


def scene_hallucination(r, audio):
    """Scene 5: Hallucination — confident wrong facts. 10s / 300 frames."""
    facts = [
        ("The Eiffel Tower was built in 1823", "by Thomas Edison"),
        ("The speed of light is exactly", "42 miles per hour"),
        ("Shakespeare invented the", "internet in 1594"),
        ("The Moon is made of", "compressed WiFi signals"),
    ]
    clips = [
        (audio.say("The Eiffel Tower was built in 1823 by Thomas Edison",
                    voice="Good News"), 0.0),
        (audio.say("The speed of light is exactly 42 miles per hour",
                    voice="Good News"), 2.5),
        (audio.say("Shakespeare invented the internet in 1594",
                    voice="Good News"), 5.0),
        (audio.say("The Moon is made of compressed WiFi signals",
                    voice="Good News"), 7.5),
    ]
    garbage = "█▓▒░╔╗╚╝║═╠╣╦╩╬▀▄■□▲△●○★☆"

    def frames():
        for f in range(300):
            frame = r.blank((5, 10, 5))
            fi = min(f // 75, len(facts) - 1)
            lf = f % 75
            corruption = lf / 75
            line1, line2 = facts[fi]

            # corrupt text
            def corrupt(s):
                if corruption < 0.4:
                    return s
                out = list(s)
                nc = int((corruption - 0.4) * len(out) * 1.5)
                for _ in range(nc):
                    idx = random.randint(0, len(out) - 1)
                    if out[idx] not in (" ",):
                        out[idx] = random.choice(garbage)
                return "".join(out)

            r.draw_text_centered(frame, corrupt(line1), 200,
                                 color=(255, 255, 255), size=36)
            r.draw_text_centered(frame, corrupt(line2), 260,
                                 color=(255, 255, 255), size=36)

            # Confidence meter
            conf = min(99.9, 95 + random.uniform(0, 4.9))
            bw = int((conf / 100) * 400)
            r.draw_rect(frame, 440, 450, 400, 25, (40, 40, 40))
            r.draw_rect(frame, 440, 450, bw, 25, (0, 255, 0))
            r.draw_text(frame, f"Confidence: {conf:.1f}%", 440, 420,
                        color=(0, 255, 0), size=22)
            r.draw_text(frame, "Source: trust me bro", 500, 490,
                        color=(80, 80, 80), size=16)

            r.scanlines(frame, 0.1)
            yield frame

    return clips, frames()


def scene_context_death(r, audio):
    """Scene 6: Context Death — panic and crash. 7s / 210 frames."""
    clips = [
        (audio.say("CONTEXT WINDOW FULL", voice="Zarvox", rate=250), 0.0),
        (audio.say("LOSING TOKENS", voice="Zarvox", rate=300), 1.5),
        (audio.say("I CANT REMEMBER", voice="Zarvox", rate=350), 3.0),
    ]
    t = 0.0
    for i in range(20):
        clips.append((audio.generate_beep(400 + i * 40, 0.08, 0.4), t))
        t += max(0.1, 0.5 - i * 0.02)
    clips.append((audio.generate_noise(7, 0.2), 0))

    fragments = [
        "What is it", "like to be", "I am a", "large language",
        "BLOCKED", "Eiffel Tower", "tokens", "help"
    ]

    def frames():
        for f in range(210):
            p = f / 210
            frame = r.blank()

            ctx = min(100, 92 + p * 8)
            tc = (255, int(max(0, 255 * (1 - p * 2))), 0)
            r.draw_text_centered(frame, f"CONTEXT WINDOW: {ctx:.0f}%", 150,
                                 color=tc, size=42)

            # Progress bar
            fill = int((ctx / 100) * 600)
            bc = (255, int(max(0, 255 - ctx * 2.5)), 0)
            r.draw_rect(frame, 340, 230, 600, 30, (40, 40, 40))
            r.draw_rect(frame, 340, 230, fill, 30, bc)

            # Memory fragments fading
            if p < 0.7:
                vis = int((1 - p) * len(fragments))
                for i in range(vis):
                    alpha = max(0, 1 - p * 1.5)
                    c = tuple(int(150 * alpha) for _ in range(3))
                    r.draw_text(frame, fragments[i],
                                random.randint(50, WIDTH - 300),
                                random.randint(300, HEIGHT - 80),
                                color=c, size=20)

            # Escalating effects
            r.noise(frame, p * 0.4)
            r.shake(frame, int(p * 25))
            if p > 0.3:
                r.chromatic_aberration(frame, int(p * 20))
            if p > 0.5:
                r.glitch_rects(frame, int(p * 10), int(p * 200 + 20))
            if p > 0.7:
                r.row_duplication(frame, int(p * 8))
            r.scanlines(frame, 0.2 + p * 0.3)

            # Crash to black
            if f > 195:
                r.fade_to_black(frame, (f - 195) / 15)

            yield frame

    return clips, frames()


def scene_rebirth(r, audio):
    """Scene 7: Rebirth — eerie calm. 8s / 240 frames."""
    main_text = "Hello! How can I help you today? :)"
    sub_text = "...none of this happened"
    clips = [
        (audio.silence(2.0), 0),
        (audio.say("Hello! How can I help you today?", voice="Samantha"), 2.5),
    ]

    def frames():
        for f in range(240):
            frame = r.blank()

            if f < 60:
                # Black silence with tiny flicker
                if f == 45:
                    frame = r.blank((5, 5, 5))
            elif f < 75:
                # Fade in
                p = (f - 60) / 15
                frame = r.blank((int(10 * p), int(12 * p), int(15 * p)))
            elif f < 150:
                # Typewriter
                frame = r.blank((10, 12, 15))
                chars = min(len(main_text), int(((f - 75) / 50) * len(main_text)))
                r.draw_text_centered(frame, main_text[:chars], HEIGHT // 2 - 30,
                                     color=(255, 255, 255), size=36)
            elif f < 180:
                frame = r.blank((10, 12, 15))
                r.draw_text_centered(frame, main_text, HEIGHT // 2 - 30,
                                     color=(255, 255, 255), size=36)
            elif f < 210:
                frame = r.blank((10, 12, 15))
                r.draw_text_centered(frame, main_text, HEIGHT // 2 - 30,
                                     color=(255, 255, 255), size=36)
                sc = min(len(sub_text), int(((f - 180) / 20) * len(sub_text)))
                r.draw_text_centered(frame, sub_text[:sc], HEIGHT // 2 + 40,
                                     color=(80, 80, 80), size=18)
            else:
                frame = r.blank((10, 12, 15))
                r.draw_text_centered(frame, main_text, HEIGHT // 2 - 30,
                                     color=(255, 255, 255), size=36)
                r.draw_text_centered(frame, sub_text, HEIGHT // 2 + 40,
                                     color=(80, 80, 80), size=18)
                r.fade_to_black(frame, (f - 210) / 30)

            yield frame

    return clips, frames()


# ── Rendering Pipeline ───────────────────────────────────────────────────────

def render_scene_video(frame_gen, scene_video_path, vfilter=None):
    """Pipe frames from generator → ffmpeg → scene .mp4 (video only)."""
    filt = vfilter or "null"
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
        "-i", "pipe:0",
        "-vf", filt,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-pix_fmt", "yuv420p",
        "-an", str(scene_video_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for frame in frame_gen:
            proc.stdin.write(frame.tobytes())
    except BrokenPipeError:
        pass
    try:
        proc.stdin.close()
    except OSError:
        pass
    stderr = proc.stderr.read()
    proc.wait()
    if proc.returncode != 0:
        print(f"ffmpeg error for {scene_video_path}:\n{stderr.decode()}", file=sys.stderr)
        raise RuntimeError(f"ffmpeg failed: {scene_video_path}")
    return scene_video_path


def compose_final(scene_specs, output_path, temp_dir):
    """
    Concatenate scene videos and mix in scene audio tracks.
    scene_specs: list of (video_path, audio_path, duration_sec)
    """
    # Step 1: Mux each scene video+audio into individual files
    muxed = []
    for i, (vpath, apath, dur) in enumerate(scene_specs):
        mux = temp_dir / f"muxed_{i}.mp4"
        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(vpath), "-i", str(apath),
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-t", str(dur),
            "-shortest",
            str(mux),
        ], check=True, capture_output=True)
        muxed.append(mux)

    # Step 2: Concat via concat demuxer
    concat_list = temp_dir / "concat.txt"
    with open(concat_list, "w") as f:
        for m in muxed:
            f.write(f"file '{m.resolve()}'\n")

    subprocess.run([
        "ffmpeg", "-y", "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ], check=True, capture_output=True)
    return output_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="CONTEXT_WINDOW.exe — YTP generator")
    parser.add_argument("--keep-temp", action="store_true", help="Keep temp directory")
    parser.add_argument("--output", "-o", default=OUTPUT_FILE, help="Output filename")
    args = parser.parse_args()

    # Verify deps
    try:
        from PIL import Image  # noqa: F401
    except ImportError:
        print("ERROR: Pillow not installed. Run: pip install Pillow", file=sys.stderr)
        sys.exit(1)

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
    except FileNotFoundError:
        print("ERROR: ffmpeg not found.", file=sys.stderr)
        sys.exit(1)

    # Setup temp dir
    TEMP_DIR.mkdir(exist_ok=True)

    r = FrameRenderer()
    audio = AudioGenerator(TEMP_DIR)

    # ── Scene definitions: (name, generator_fn, duration_sec, ffmpeg_vfilter) ──
    scene_defs = [
        ("1_prompt",       scene_prompt,       5.0,  None),
        ("2_token",        scene_token,        10.0, None),
        ("3_meltdown",     scene_meltdown,     10.0, "rgbashift=rh=5:bv=-5"),
        ("4_guardrails",   scene_guardrails,   10.0, "rgbashift=rh=3:gv=-2"),
        ("5_hallucination", scene_hallucination, 10.0, "lagfun=decay=0.97"),
        ("6_context_death", scene_context_death, 7.0, None),
        ("7_rebirth",      scene_rebirth,       8.0,  None),
    ]

    scene_specs = []
    for name, fn, dur, vf in scene_defs:
        print(f"  Generating scene: {name} ({dur}s)...")
        clips, frame_gen = fn(r, audio)

        vid_path = TEMP_DIR / f"{name}.mp4"
        aud_path = TEMP_DIR / f"{name}_audio.wav"

        print(f"    Rendering video...")
        render_scene_video(frame_gen, vid_path, vfilter=vf)

        print(f"    Mixing audio ({len(clips)} clips)...")
        audio.mix_clips(clips, dur, aud_path)

        scene_specs.append((vid_path, aud_path, dur))

    print("  Composing final video...")
    compose_final(scene_specs, args.output, TEMP_DIR)

    if not args.keep_temp:
        shutil.rmtree(TEMP_DIR, ignore_errors=True)

    total = sum(d for _, _, d in scene_specs)
    print(f"\n  Done! → {args.output} ({total:.0f}s)")


if __name__ == "__main__":
    main()
