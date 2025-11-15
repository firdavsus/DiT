from itertools import islice
import os
import json
import asyncio
from io import BytesIO
from PIL import Image
from datasets import load_dataset
from tqdm.asyncio import tqdm
import aiohttp
import aiofiles
import async_timeout

# ---------------- CONFIG ----------------
output_folder = "all_images"
captions_file = "captions.json"
os.makedirs(output_folder, exist_ok=True)

dataset_split = "train"      # or "validation" / "test"
max_concurrent = 128        # max simultaneous connections
chunk_size = 1024             # process dataset in chunks
timeout = 4                   # seconds per request
img_size = 256
save_every = 25            # save captions periodically

# ---------------- helpers ----------------
async def fetch_image_aio(session, url, size=img_size, timeout=timeout):
    """Download and center-crop+resize image asynchronously."""
    if not url:
        return None
    try:
        async with async_timeout.timeout(timeout):
            async with session.get(url, headers={"User-Agent":"Mozilla/5.0"}) as resp:
                if resp.status != 200:
                    return None
                content = await resp.read()
                im = Image.open(BytesIO(content)).convert("RGB")
                W, H = im.size
                m = min(W, H)
                im = im.crop(((W - m)//2, (H - m)//2, (W + m)//2, (H + m)//2)).resize((size, size), Image.BICUBIC)
                return im
    except Exception:
        return None

async def download_worker(session, idx, ex):
    """Download single image and return (idx, caption) or None."""
    url = ex.get("image_url")
    caption = ex.get("caption", "")
    if not url:
        return None
    img = await fetch_image_aio(session, url)
    if img is None:
        return None
    tmp_path = os.path.join(output_folder, f"{idx}.tmp")
    final_path = os.path.join(output_folder, f"{idx}.jpg")
    try:
        img.save(tmp_path, format="JPEG", quality=92)
        os.replace(tmp_path, final_path)
        return idx, caption
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return None

def chunked_iterable(it, size):
    it = iter(it)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk

# ---------------- resume support ----------------
captions = {}
downloaded_ids = set()

# 1. Load existing captions if any
if os.path.exists(captions_file):
    try:
        with open(captions_file, "r", encoding="utf-8") as f:
            captions = json.load(f)
        downloaded_ids = set(int(k) for k in captions.keys())
        print(f"Resuming: loaded {len(captions)} captions")
    except Exception:
        captions = {}
        downloaded_ids = set()

# 2. Scan image folder and detect images without captions
existing_files = [f for f in os.listdir(output_folder) if f.endswith(".jpg")]
existing_ids = set(int(os.path.splitext(f)[0]) for f in existing_files)

# Add missing IDs into captions dict
missing_ids = existing_ids - downloaded_ids
if missing_ids:
    print(f"Found {len(missing_ids)} images missing in captions.json – adding blank captions.")
    for mid in missing_ids:
        captions[str(mid)] = ""
    downloaded_ids |= missing_ids

# ---------------- main async downloader ----------------
dataset = load_dataset("conceptual_captions", split=dataset_split)
dataset_iter = iter(dataset)
try:
    total_estimate = len(dataset)
except Exception:
    total_estimate = None

async def main():
    global captions, downloaded_ids
    global_idx = 0
    pbar = tqdm(total=total_estimate, desc="Attempted", unit="img")
    pbar.set_postfix(saved=len(captions), refresh=False)

    connector = aiohttp.TCPConnector(limit=max_concurrent, ssl=False)
    timeout_obj = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout_obj) as session:
        for chunk in chunked_iterable(dataset_iter, chunk_size):
            tasks = []
            for i, ex in enumerate(chunk):
                idx = global_idx + i
                if idx in downloaded_ids:
                    pbar.update(1)
                    continue
                tasks.append(download_worker(session, idx, ex))

            for fut in asyncio.as_completed(tasks):
                res = await fut
                pbar.update(1)
                if res is not None:
                    idx_saved, caption = res
                    captions[str(idx_saved)] = caption
                    downloaded_ids.add(idx_saved)
                if len(captions) % 100 == 0:
                    pbar.set_postfix(saved=len(captions), refresh=True)

            global_idx += len(chunk)

            # periodically save captions
            if len(captions) > 0 and len(captions) % save_every == 0:
                print("it is zsaved")
                async with aiofiles.open(captions_file, "w", encoding="utf-8") as f:
                    await f.write(json.dumps(captions, ensure_ascii=False))
                pbar.set_postfix(saved=len(captions), refresh=True)

    # final save
    if len(captions) > 0:
        async with aiofiles.open(captions_file, "w", encoding="utf-8") as f:
            await f.write(json.dumps(captions, ensure_ascii=False))
    pbar.close()
    print("Done. saved images with captions:", len(captions))

# ---------------- run ----------------
asyncio.run(main())