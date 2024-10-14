import os
import asyncio
import websockets
import json
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

async def connect_to_openai_websocket(audio_event):
    api_key = os.getenv('OPENAI_API_KEY')

    if not api_key:
        print("API Key not found! Exiting connection attempt.")
        return

    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "OpenAI-Beta": "realtime=v1"
    }
    
    try:
        async with websockets.connect(url, extra_headers=headers) as ws:
            print("Connected to server.")
            await ws.send(audio_event)
            print("Audio event sent.")

            response_message = {
                "type": "response.create",
                "response": {
                    "modalities": ["text"],
                    "instructions": "Please respond in a friendly manner."
                }
            }
            await ws.send(json.dumps(response_message))
            print("Response create command sent.")

            async for message in ws:
                event = json.loads(message)
                print("Received message:", event)

    except websockets.exceptions.InvalidStatusCode as e:
        print(f"Connection failed with status code: {e.status_code}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

asyncio.run(connect_to_openai_websocket("Test Audio Event"))