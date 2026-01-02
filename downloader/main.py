import requests, uuid, mimetypes, json
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import os

load_dotenv()

DISCORD_AUTHORIZATION_HEADER = os.getenv("DISCORD_AUTHORIZATION_HEADER")

LAST_MESSAGE_ID = 1379218891192012983
MESSAGE_BUFFER_CAPACITY = 10000
MAX_WORKERS = 32

pid = 9223372036854775807
message_buffer = []
buffers = 0

def download_media(url, name, retries=10):
    for attempt in range(1, retries + 1):
        try:
            media = requests.get(url, timeout=20)
            if media.status_code != 200:
                print(f"Attempt {attempt}: Bad status {media.status_code} for {url}")
                continue

            content_type = media.headers.get("Content-Type")
            ext = mimetypes.guess_extension(content_type) or ".bin"

            filepath = Path(f"downloader/media/{name}{ext}")
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, "wb") as f:
                f.write(media.content)

            return f"{name}{ext}"  # return full filename
        except Exception as e:
            print(f"Attempt {attempt} failed for {url}: {e}")

    print(f"Failed to download {url} after {retries} attempts")
    return None

def download_message_media(message):
    media_files = []

    # attachments
    for attachment in message["attachments"]:
        name = str(uuid.uuid4())
        media_files.append((attachment["url"], name))
    
    # embeds
    for embed in message.get("embeds", []):
        if embed.get("type") == "image" and "url" in embed:
            name = str(uuid.uuid4())
            media_files.append((embed["url"], name))
    
    return media_files

while True:
    response = requests.get(f"https://discord.com/api/v9/channels/535250426061258753/messages?before={pid}&limit=100", headers={
        "Authorization": DISCORD_AUTHORIZATION_HEADER
    })

    if response.status_code == 200:
        messages = response.json()

        if len(messages) == 0:
            break

        pid = int(messages[-1]["id"])

        print(messages[0]["timestamp"])

        if pid <= LAST_MESSAGE_ID:
            break

        all_media = []
        media_mapping = {}  # map (url -> name) so we can associate files with messages

        for message in messages:
            files = download_message_media(message)
            for url, name in files:
                media_mapping[url] = name
                all_media.append((url, name))

        # Download in parallel
        # Download in parallel and get filenames
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_url = {executor.submit(download_media, url, name): url for url, name in all_media}
            downloaded_files = {}  # url -> full filename
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                filename = future.result()
                if filename:  # only add if download succeeded
                    downloaded_files[url] = filename

        # Now build buffer
        for message in messages:
            media_files = []

            for attachment in message.get("attachments", []):
                url = attachment["url"]
                if url in downloaded_files:
                    media_files.append(downloaded_files[url])

            for embed in message.get("embeds", []):
                if embed.get("type") == "image" and "url" in embed:
                    url = embed["url"]
                    if url in downloaded_files:
                        media_files.append(downloaded_files[url])

            if media_files:
                message_buffer.append([
                    message.get("content", ""),
                    datetime.fromisoformat(message["timestamp"]).timestamp(),
                    media_files
                ])

        if len(message_buffer) >= MESSAGE_BUFFER_CAPACITY:
            with open(f"downloader/buffers/{buffers}.json", "w") as f:
                f.write(json.dumps(message_buffer, separators=(',', ':')))
            
            message_buffer = []
            buffers += 1

    else:
        print(response.status_code, response.text)

if message_buffer:
    with open(f"downloader/buffers/{buffers}.json", "w") as f:
        f.write(json.dumps(message_buffer, separators=(',', ':')))
