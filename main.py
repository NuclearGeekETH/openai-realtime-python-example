import os
import io
import json
import asyncio
import base64
import websockets
from pydub import AudioSegment
import soundfile as sf
import gradio as gr
from dotenv import load_dotenv

load_dotenv()

async def connect_to_openai_websocket(audio_event):
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}",
        "OpenAI-Beta": "realtime=v1",
    }

    async with websockets.connect(url, extra_headers=headers) as ws:
        print("Connected to server.")

        # Send audio event to the server
        await ws.send(audio_event)
        print("Audio event sent.")

        async for message in ws:
            event = json.loads(message)

            # Check if the message is an audio response
            if event.get('type') == 'conversation.item.created':

                # Send a command to create a response
                response_message = {
                    "type": "response.create"
                }
                await ws.send(json.dumps(response_message))
                print("Response create command sent.")

                audio_data_list = []

                # Listen for messages from the server
                async for message in ws:
                    event = json.loads(message)

                    # Check if the message is an audio response
                    if event.get('type') == 'response.audio.delta':
                        audio_data_list.append(event['delta'])

                    # Check if the message is an audio response
                    if event.get('type') == 'response.audio.done':
                        full_audio_base64 = ''.join(audio_data_list)  

                        audio_data = base64.b64decode(full_audio_base64)
                        return audio_data

def numpy_to_audio_bytes(audio_np, sample_rate):
    with io.BytesIO() as buffer:
        # Write the audio data to the buffer in WAV format
        sf.write(buffer, audio_np, samplerate=sample_rate, format='WAV')
        buffer.seek(0)  # Move to the beginning of the buffer
        wav_bytes = buffer.read()
    return wav_bytes

def audio_to_item_create_event(audio_data: tuple) -> str:
    sample_rate, audio_np = audio_data
    audio_bytes = numpy_to_audio_bytes(audio_np, sample_rate)
    
    pcm_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    event = {
        "type": "conversation.item.create",
        "item": {
            "type": "message",
            "role": "user",
            "content": [{
                "type": "input_audio",
                "audio": pcm_base64
            }]
        }
    }
    return json.dumps(event)

def voice_chat_response(audio_data, history):
    audio_event = audio_to_item_create_event(audio_data)
    audio_response = asyncio.run(connect_to_openai_websocket(audio_event))

    if isinstance(audio_response, bytes):
        audio_io = io.BytesIO(audio_response)
        audio_segment = AudioSegment.from_raw(
            audio_io, 
            sample_width=2, 
            frame_rate=24000, 
            channels=1
        )
        
        # Output audio as file-compatible stream for Gradio playback
        with io.BytesIO() as buffered:
            audio_segment.export(buffered, format="wav")
            return buffered.getvalue(), history  #

    return None, history

# Gradio Interface Setup
with gr.Blocks(title="OpenAI Realtime API") as demo:
    gr.Markdown("<h1 style='text-align: center;'>OpenAI Realtime API</h1>")

    with gr.Tab("VoiceChat"):
        gr.Markdown("Speak to interact with the OpenAI model in real-time and hear its responses.")

        audio_input = gr.Audio(
            label="Record your voice",
            sources="microphone",
            type="numpy",
            render=True
        )
        
        audio_output = gr.Audio(
            autoplay=True,
            render=True
        )
        
        history_state = gr.State([])

        gr.Interface(
            fn=voice_chat_response,
            inputs=[audio_input, history_state],
            outputs=[audio_output, history_state]
        )

if __name__ == "__main__":
    demo.launch()