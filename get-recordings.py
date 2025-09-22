#!/usr/bin/env python3
"""
Manual script to check for Twilio recordings from the recent call
"""

import os
from dotenv import load_dotenv
from twilio.rest import Client
from datetime import datetime, timedelta, timezone

load_dotenv()

ACCT_SID = os.getenv("TWILIO_ACCOUNT_SID")
AUTH_TOK = os.getenv("TWILIO_AUTH_TOKEN")

def check_recordings():
    client = Client(ACCT_SID, AUTH_TOK)

    # Create recordings directory if it doesn't exist
    os.makedirs("recordings", exist_ok=True)

    # Check for recordings in the last hour
    recent_time = datetime.now(timezone.utc) - timedelta(hours=1)
    recordings = client.recordings.list(date_created_after=recent_time)

    print(f"Found {len(recordings)} recordings in the last hour:")

    for i, recording in enumerate(recordings):
        mp3_url = f"https://api.twilio.com{recording.uri.replace('.json', '.mp3')}"
        print(f"\n--- Recording {i+1} ---")
        print(f"SID: {recording.sid}")
        print(f"Call SID: {recording.call_sid}")
        print(f"Duration: {recording.duration} seconds")
        print(f"Created: {recording.date_created}")
        print(f"Status: {recording.status}")
        print(f"Channels: {recording.channels}")
        print(f"MP3 URL: {mp3_url}")

        # Try to download this one
        if recording.status == 'completed':
            import requests
            filename = f"recordings/recording_{recording.sid}.mp3"

            # Check if file already exists
            if os.path.exists(filename):
                print(f"Already exists: {filename}")
                continue

            try:
                auth = (ACCT_SID, AUTH_TOK)
                response = requests.get(mp3_url, auth=auth, stream=True, timeout=60)
                response.raise_for_status()  # Raise exception for bad status codes

                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print(f"✅ Downloaded: {filename}")

            except requests.exceptions.RequestException as e:
                print(f"Download failed: {e}")
        else:
            print(f"⏳ Status: {recording.status} (not ready for download)")

    if not recordings:
        print("No recordings found. Possible reasons:")
        print("1. Recording still processing (wait a few minutes)")
        print("2. Recording failed to start")
        print("3. Call SID mismatch")

if __name__ == "__main__":
    check_recordings()
