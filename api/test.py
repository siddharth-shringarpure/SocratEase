import os
from pyneuphonic import Neuphonic, TTSConfig
from pyneuphonic.player import AudioPlayer

# Load the API key from the environment
client = Neuphonic(api_key='49b59869a21b9499f55cd4d3bc8680b147d5f1bca42b06e71e204b82bfcbd016.f52b43ef-edbd-4cd5-909c-8535c1ce2591')
voices = client.voices.list()
print("testing")
for voice in voices:
    print(voice)
    print(type(voice))
# sse = client.tts.SSEClient()

# # TTSConfig is a pydantic model so check out the source code for all valid options
# tts_config = TTSConfig(
#     speed=1.05,
#     lang_code='en', # replace the lang_code with the desired language code.
#     voice_id='e564ba7e-aa8d-46a2-96a8-8dffedade48f'  # use client.voices.list() to view all available voices
# )

# # Create an audio player with `pyaudio`
# with AudioPlayer() as player:
#     response = sse.send('Hello, world!', tts_config=tts_config)
#     player.play(response)

#     player.save_audio('output.wav')  # save the audio to a .wav file