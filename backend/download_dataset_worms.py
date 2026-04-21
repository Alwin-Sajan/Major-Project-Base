import os
import requests
import time
import shutil
import asyncio
import aiohttp
import aiofiles

# --- CONFIGURATION ---
BASE_PATH = "/media/abk/New Disk/DATASETS/marine_v2"
TRAIN_COUNT = 500
VAL_COUNT = 50
TOTAL_PER_SPECIES = TRAIN_COUNT + VAL_COUNT
BATCH_SIZE = 100 # GBIF API limit per request
SLEEP_TIME = 1.5  # Seconds between API calls to avoid rate limits

# --- TAXONOMY DATA ---
# Structure: { Species_Name: wormsid } 
dataset_map = {
    "sungami" : 209270,
    "neglecta" : 475003
}


#------------------ ASYNC -------------------------------------------------------------------------

async def get_nub_key_async(session:aiohttp.ClientSession, aphia_id:int) -> str:
    WORMS_DATASET_KEY = "2d59e5db-57ad-41ff-97d6-11f5fb264527"
    lsid = f"urn:lsid:marinespecies.org:taxname:{aphia_id}"
    species_api_url = f"https://api.gbif.org/v1/species"
    params = {
        "datasetKey" : WORMS_DATASET_KEY,
        "sourceId" : lsid
    }
    async with session.get(species_api_url, params=params) as response:
        response.raise_for_status()
        data = await response.json()
        
        results = data.get("results", [])
        if not results:
            print("No result for aphia id:",aphia_id)
            return
        
        first_res:dict = results[0]
        nub_key = first_res.get('nubKey')
        family = first_res.get('family')
        genus = first_res.get('genus')
        canonical_name = first_res.get('canonicalName')
        res = {
            "nubKey": nub_key,
            "family": family,
            "genus": genus,
            "scientificName": canonical_name
        }

        if missing := [name for name, value in res.items() if value is None]:
            print(f"{canonical_name} - Couldn't fetch data of {', '.join(missing)}")
            return 

        print(f"Name: {canonical_name} - Nubkey: {nub_key}")
        return res


async def download_image_async(session:aiohttp.ClientSession, download_url:str, save_path:str):
    if os.path.exists(save_path): 
        return True
    try:
        async with session.get(download_url, timeout=60) as response:
            response.raise_for_status()
            async with aiofiles.open(save_path, 'wb') as f:
                async for chunk in response.content.iter_chunked(1024):
                    await f.write(chunk)
            return True
    except:
        return False


async def process_species_async(preferred_species_name:str, wormsid:int):

    total_assigned = 0  # Track how many we've decided to download
    downloaded_count = 0
    offset = 0
    semaphore = asyncio.Semaphore(20)  # Limit concurrent downloads
    connector = aiohttp.TCPConnector(limit=50)

    async with aiohttp.ClientSession(connector=connector) as session:
        tax_data = await get_nub_key_async(session,wormsid)
        if not tax_data: 
            return
        
        f_name = tax_data['family']
        g_name = tax_data['genus']
        nub_key = tax_data['nubKey']
        print(f"\n[START] Processing: {f_name} > {g_name} > {preferred_species_name} (ID: {wormsid})")
            # Create Directories
        train_dir = os.path.join(BASE_PATH, "train", f_name, g_name, preferred_species_name)
        val_dir = os.path.join(BASE_PATH, "val", f_name, g_name, preferred_species_name)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        while total_assigned < TOTAL_PER_SPECIES:

            api_url = "https://api.gbif.org/v1/occurrence/search"
            params = {
                "taxonKey": nub_key,
                "mediaType": "StillImage",
                "limit": BATCH_SIZE,
                "offset": offset
            }
            
            try:
                async with session.get(api_url, params=params) as response:
                    response.raise_for_status()
                    data = await response.json()

                    results = data.get("results", [])

                    if not results: 
                        break 

                tasks = []
                
                for record in results:

                    if total_assigned >= TOTAL_PER_SPECIES: 
                        break

                    media = [m for m in record.get("media", []) if m.get("type") == "StillImage"]
                    if not media: 
                        continue
                    
                    img_url = media[0].get("identifier")
                    if not img_url:
                        continue

                    target_dir = train_dir if total_assigned < TRAIN_COUNT else val_dir
                    file_name = f"{wormsid}_{total_assigned}_{int(time.time())}.jpg"
                    save_path = os.path.join(target_dir, file_name)
                    total_assigned += 1

                    async def sem_download(url, path):
                        async with semaphore:
                            return await download_image_async(session, url, path)
                    
                    tasks.append(asyncio.create_task(sem_download(img_url, save_path)))

                if tasks:
                    results = await asyncio.gather(*tasks)
                    downloaded_count += sum(1 for r in results if r)
                    print(f"   {preferred_species_name}: Progress {total_assigned}/{TOTAL_PER_SPECIES} assigned...")
                                   
                if data.get("endOfRecords"): 
                    break

                offset += BATCH_SIZE
                await asyncio.sleep(SLEEP_TIME)
                

            except Exception as e:
                print(f"   Error fetching {preferred_species_name}: {e}")
                break

    print(f"[FINISH] {preferred_species_name}: Total {downloaded_count} images.")


async def run_pipeline_async():
    print(f"Initializing Marine_v2 Dataset at {BASE_PATH}...")

    semaphore = asyncio.Semaphore(3)

    async def sem_tasks(species_name, taxon_key):
        async with semaphore:
            await process_species_async(species_name, taxon_key)
    
    tasks = [asyncio.create_task(sem_tasks(name, wid)) for name, wid in dataset_map.items()]
    await asyncio.gather(*tasks)

#---------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(run_pipeline_async())