from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

client = ElevenLabs(
    api_key="a7e2a4013f7a2c0bdad2406427b1eebf64a1d2ed3af3ffc0136096c9ce0ed6cb"
)

audio = client.text_to_speech.convert(
    text="hola como estas como te llamas.",
    voice_id="kcQkGnn0HAT2JRDQ4Ljp",
    model_id="eleven_multilingual_v2",
    output_format="mp3_44100_128",
)

play(audio)