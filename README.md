Python bindings for whisper.cpp
===============================

`pip install git+https://github.com/YannisTevissen/whisper-ggml-python`

```python
from whispercpp import Whisper

w = Whisper('tiny')

result = w.transcribe("myfile.mp3", lang="en")
text = w.extract_text(result)
```

Note: default parameters might need to be tweaked.
See Whispercpp.pyx.
