import asyncio
import websockets
import json

async def test_connection():
    uri = "ws://192.168.1.3:5001"
    async with websockets.connect(uri) as websocket:
        # Wait for the ready message
        message = await websocket.recv()
        print(f"Received: {message}")
        
        # Send a test message   
        test_message = json.dumps({"input": "Explain newton's third law"})
        await websocket.send(test_message)
        
        # Receive the response
        response = await websocket.recv()
        print(f"Response: {response}")

asyncio.run(test_connection())