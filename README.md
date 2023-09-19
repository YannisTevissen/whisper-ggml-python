Python bindings for whisper.cpp
===============================

- Run `pip install git+https://github.com/YannisTevissen/whisper-ggml-python`
- set `WHISPER_MODELS`as the directory where the models will be stored

```python
from whispercpp import Whisper

w = Whisper('tiny')

result = w.transcribe("myfile.mp3", lang="en")
text = w.extract_text(result)
```

Note: default parameters might need to be tweaked.
See Whispercpp.pyx.
