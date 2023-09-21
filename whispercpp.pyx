#!python
# cython: language_level=3

# import ffmpeg
import numpy as np
import requests
import os, subprocess
from pathlib import Path

MODELS_DIR = os.environ['WHISPER_MODELS']
print("Saving models to:", MODELS_DIR)


cimport numpy as cnp

cdef int SAMPLE_RATE = 16000
cdef char* TEST_FILE = 'test.wav'
cdef char* DEFAULT_MODEL = 'tiny'
cdef int N_THREADS = os.cpu_count()

MODELS = {
    'ggml-tiny.bin': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
    'ggml-base.bin': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
    'ggml-small.bin': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin',
    'ggml-medium.bin': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin',
    'ggml-large.bin': 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large.bin',
}

def model_exists(model):
    return os.path.exists(Path(MODELS_DIR).joinpath(model))

def download_model(model):
    if model_exists(model):
        return

    print(f'Downloading {model}...')
    url = MODELS[model]
    r = requests.get(url, allow_redirects=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(Path(MODELS_DIR).joinpath(model), 'wb') as f:
        f.write(r.content)


cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] load_audio(str file, int sr = SAMPLE_RATE):
    try:
        print(os.path.exists("./ffmpeg"))
        print(os.path.exists("ffmpeg"))
        print(os.path.exists(file))
        command = [
            "./ffmpeg",
            "-y",
            "-loglevel", "debug",
            "-i", file,
            "-f", "s16le",
            "-acodec", "pcm_s16le",
            "-ac", "1",
            "-ar", str(sr),
            "-nostdin",
            "-"
        ]

        # Run the FFmpeg command and capture stdout and stderr
        # Run the FFmpeg command and capture stdout and stderr
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        print(stderr.decode('utf-8'))
        # Check for errors
        if process.returncode != 0:
            print(f"Error: {stderr.decode('utf-8')}")
        else:
            out = stdout
    except:
        raise RuntimeError(f"File '{file}' not found")
    #print(out)
    cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = (
        np.frombuffer(out, np.int16)
        .flatten()
        .astype(np.float32)
    ) / pow(2, 15)
    #print(frames)
    return frames

cdef whisper_full_params default_params(char* language) nogil:
    cdef whisper_full_params params = whisper_full_default_params(
        whisper_sampling_strategy.WHISPER_SAMPLING_GREEDY
    )
    params.print_realtime = True
    params.print_progress = True
    params.translate = False
    params.language = <const char *>language
    n_threads = N_THREADS
    return params


cdef class Whisper:
    cdef whisper_context * ctx
    cdef whisper_full_params params

    def __init__(self, model=DEFAULT_MODEL, pb=None, buf=None):

        if not os.path.exists(model):
            print('No valid path given')
            model_fullname = f'ggml-{model}.bin'
            download_model(model_fullname)
            model_path = Path(MODELS_DIR).joinpath(model_fullname)
        else:
            model_path = model
        cdef bytes model_b = str(model_path).encode('utf8')
        
        if buf is not None:
            self.ctx = whisper_init_from_buffer(buf, buf.size)
        else:
            self.ctx = whisper_init_from_file(model_b)
        
        whisper_print_system_info()


    def __dealloc__(self):
        whisper_free(self.ctx)

    def transcribe(self, filename=TEST_FILE, lang="auto"):
        print("Loading data..")
        if (type(filename) == np.ndarray) :
            temp = filename
        
        elif (type(filename) == str) :
            temp = load_audio(<str>filename)
        else :
            temp = load_audio(<str>TEST_FILE)

        
        cdef cnp.ndarray[cnp.float32_t, ndim=1, mode="c"] frames = temp
        language_bytes = str(lang).encode("utf-8")
        cdef char* c_string = language_bytes  # Convert bytes to char*

        params = default_params(c_string)

        print("Transcribing..")

        return whisper_full(self.ctx, params, &frames[0], len(frames))
    
    def extract_text(self, int res):
        print("Extracting text...")
        if res != 0:
            raise RuntimeError
        cdef int n_segments = whisper_full_n_segments(self.ctx)
        return [
            whisper_full_get_segment_text(self.ctx, i).decode() for i in range(n_segments)
        ]


