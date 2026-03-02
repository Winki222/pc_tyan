import sounddevice as sd
import soundfile as sf
import numpy as np
import os, random, time, sys, threading, queue

try:
    from pydub import AudioSegment
    PYDUB = True
except:
    PYDUB = False
INPUT_DEVICE  = 1       #индекс микрофона (смотри список при запуске)
OUTPUT_DEVICE = 6       #индекс динамиков,наушников индивидуально под ноут
FOLDER        = "moan"  #папка со звуками
THRESHOLD     = 0.02    #порог срабатывания (0.01 тихо / 0.05 громко)
MAX_SECS      = 3.0     #обрезать звук до N секунд
COOLDOWN      = 7.0     #пауза между срабатываниями 

def load(path):
    try:
        d, sr = sf.read(path, dtype='float32')
        return (d.mean(axis=1) if d.ndim == 2 else d), sr
    except Exception:
        if not PYDUB:
            raise
        seg = AudioSegment.from_file(path)
        arr = np.array(seg.get_array_of_samples()).astype(np.float32)
        if seg.channels > 1:
            arr = arr.reshape((-1, seg.channels)).mean(axis=1)
        return arr / (32768.0 if arr.max() > 1 else 1.0), seg.frame_rate

def load_sounds(folder):
    exts = ('.wav', '.mp3', '.ogg', '.flac', '.aif', '.aiff')
    sounds = []
    for f in sorted(os.listdir(folder)):
        if not f.lower().endswith(exts):
            continue
        try:
            d, sr = load(os.path.join(folder, f))
            d = d[:int(MAX_SECS * sr)]
            sounds.append((d, sr, f))
            print(f"   {f}  ({len(d)/sr:.1f}s)")
        except Exception as e:
            print(f"   {f}: {e}")
    return sounds
triggered  = False
last_play  = 0.0
is_playing = False
lock       = threading.Lock()
q: queue.Queue = queue.Queue(maxsize=1)

def callback(indata, frames, t, status):
    global triggered
    if is_playing:         
        triggered = False
        return
    rms = float(np.sqrt(np.mean(indata ** 2)))
    if rms >= THRESHOLD:
        if not triggered:
            try: q.put_nowait(rms)
            except queue.Full: pass
            triggered = True
    else:
        triggered = False
def play(snd):
    def _run():
        global is_playing
        with lock:
            is_playing = True
            try:
                sd.play(snd[0], snd[1], device=OUTPUT_DEVICE, blocking=True)
            except Exception as e:
                print(f" Ошибка {e}")
            finally:
                time.sleep(0.5)   
                is_playing = False
    threading.Thread(target=_run, daemon=True).start()
def main():
    global last_play
    sounds = load_sounds(FOLDER)
    if not sounds:
        sys.exit("Нет стонов, вы гей!")

    sr_in     = int(sd.query_devices(INPUT_DEVICE)['default'])
    blocksize = int(sr_in * 0.05)

    stream = sd.InputStream(
        device=INPUT_DEVICE, channels=1,
        samplerate=sr_in, blocksize=blocksize,
        dtype='float32', callback=callback
    )

    print(f"\n Шлепни меняя!\n{'─'*40}\n")
    stream.start()

    try:
        while True:
            try:
                rms = q.get(timeout=0.1)
            except queue.Empty:
                continue

            now = time.time()
            if is_playing or (now - last_play) < COOLDOWN:
                continue

            snd = random.choice(sounds)
            print(f"  [{time.strftime('%H:%M:%S')}]  {snd[2]}  (rms={rms:.4f})")
            play(snd)
            last_play = now

    except KeyboardInterrupt:
        print("\n  пока пока , извращенец!")
    finally:
        stream.stop()
        stream.close()
        sd.stop()

if __name__ == "__main__":
    main()
