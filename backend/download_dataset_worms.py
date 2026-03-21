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
# Structure: { Family: { Genus: { Species_Name: wormsid } } }
dataset_map = {
    "Acanthuridae": {
        # "Ctenochaetus": {
        #     "striatus": 219659,       # Striated Surgeonfish (Extremely abundant)
        #     "strigosus": 219658,      # Goldring Bristletooth
        #     "binotatus": 219655,      # Twospot Bristletooth
        #     "tominiensis": 277562,    # Tomini Surgeonfish
        #     "truncatus": 277563,      # Indian Gold-ring Bristletooth
        #     "hawaiiensis": 277561,    # Chevron Tang
        #     "cyanocheilus": 277559,   # Bluelip Bristletooth
        #     "flavicauda": 277560      # Whitetail Bristletooth
        # },
        # "Zebrasoma": {
        #     "flavescens": 219683,     # Yellow Tang (Massive image volume)
        #     "scopas": 219679,         # Brushtail Tang
        #     #"veliferum": 219683,      # Sailfin Tang
        #     "desjardinii": 277555,    # Red Sea Sailfin Tang
        #     "xanthurum": 219681,      # Yellowtail Blue Tang
        #     #"rostratum": 277556,      # Longnose Tang
        #     #"gemmatum": 219680,       # Gem Tang
        # },
        "Paracanthurus": {
            "hepatus": 219676          # Palette Surgeonfish / Blue Tang
        }
        # "Acanthurus": {
        #     "triostegus": 219630,      # Convict Surgeonfish
        #     "lineatus": 159582,       # Lined Surgeonfish
        #     "nigrofuscus": 219640,    # Brown Surgeonfish
        #     "leucosternon": 219628,   # Powder Blue Tang
        #     "dussumieri": 219641,     # Eyestripe Surgeonfish
        #     "xanthopterus": 219634,   # Yellowfin Surgeonfish
        #     "olivaceus": 219625,      # Orange-band Surgeonfish
        #     "pyroferus": 219648       # Chocolate Surgeonfish
        # },
        # "Naso": {
        #     "lituratus": 219665,      # Orangespine Unicornfish
        #     "unicornis": 219668,      # Bluespine Unicornfish
        #     "vlamingii": 219672,      # Bignose Unicornfish
        #     #"annulatus": 219673,      # Whitemargin Unicornfish
        #     "hexacanthus": 219667,    # Sleek Unicornfish
        #     "brachycentron": 219664,  # Humpback Unicornfish
        #     "brevirostris": 219671,   # Spotted Unicornfish
        #     #"tonganus": 278010        # Bulbnose Unicornfish
        # }
    },
    # "Chaetodontidae": {
    #     "Chaetodon": {
    #         "auriga": 218730,         # Threadfin Butterflyfish
    #         "lunula": 218733,         # Raccoon Butterflyfish
    #         "kleinii": 218738,        # Sunburst Butterflyfish
    #         #"vagabundus": 218765,     # Vagabond Butterflyfish
    #         #"lineolatus": 218749,     # Lined Butterflyfish
    #         "trifascialis": 218719,   # Chevron Butterflyfish
    #         "melannotus": 218743,     # Black-back Butterflyfish
    #         #"ephippium": 218740       # Saddle Butterflyfish
    #     },
    #     "Heniochus": {
    #         "acuminatus": 218765,     # Longfin Bannerfish
    #         "diphreutes": 218763,     # Schooling Bannerfish
    #         "monoceros": 218764,      # Masked Bannerfish
    #         "chrysostomus": 276748,   # Pennant Bannerfish
    #         "varius": 276750,         # Humphead Bannerfish
    #         "singularius": 218766,    # Singular Bannerfish
    #         "intermedius": 218767,    # Red Sea Bannerfish
    #         "pleurotaenia": 276749    # Phantom Bannerfish
    #     }
    # }
}

    # "Gobiidae": {
    #     "Amblyeleotris": {
    #         "aurora": 209272, "diagonalis": 209269,"wheeleri": 209274, "latifasciata" : 278972, 
    #         "randalli" : 278980, "fontanesii" : 278966, "yanoi" : 278985, "steinitzi" : 209273
    #     },
    #     "Gobiodon": {
    #         "okinawae": 276931, "histrio": 276928, "citrinus": 5208803, "rivulatus": 2378622,
    #         "quinquestrigatus": 276933, "erythrospilus": 309999, "ceramensis": 276926, "axillaris": 276924
    #     }
    # },
    # "Serranidae": {
    #     "Epinephelus": {
    #         "fasciatus": 2388344, "merra": 2388271, "coioides": 2388213, "striatus": 2388300,
    #         "guttatus": 2388558, "marginatus": 2388507, "polyphekadion": 2388423, "diacanthus": 2388126
    #     },
    #     "Cephalopholis": {
    #         "argus": 2388891, "miniata": 2381788, "urodeta": 2388881,
    #         "sonnerati": 2388907, "cruentata": 2388903, "leoparda": 2388887, "taeniops": 2388910
    #     }
    # }



def get_nub_key(aphia_id:int) -> str:
    WORMS_DATASET_KEY = "2d59e5db-57ad-41ff-97d6-11f5fb264527"
    lsid = f"urn:lsid:marinespecies.org:taxname:{aphia_id}"

    species_url = f"https://api.gbif.org/v1/species?datasetKey={WORMS_DATASET_KEY}&sourceId={lsid}"
    species_res = requests.get(species_url).json()

    if not species_res.get("result"):
        print("No result for aphia id:",aphia_id)
        return
    
    nub_key = species_res['results'][0].get('nubKey')
    canonical_name = species_res['results'][0].get('canonicalName')
    print(f"Name: {canonical_name} - Nubkey: {nub_key}")
    return nub_key



def download_image(url, save_path):
    try:
        if os.path.exists(save_path): return True
        r = requests.get(url, stream=True, timeout=10)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)
            return True
    except:
        pass
    return False




def process_species(f_name:str, g_name:str, s_name:str, taxon_key:int):
    print(f"\n[START] Processing: {f_name} > {g_name} > {s_name} (ID: {taxon_key})")
    # Create Directories
    train_dir = os.path.join(BASE_PATH, "train", f_name, g_name, s_name)
    val_dir = os.path.join(BASE_PATH, "val", f_name, g_name, s_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    downloaded = 0
    offset = 0
    
    while downloaded < TOTAL_PER_SPECIES:
        nub_key = get_nub_key(taxon_key)
        api_url = "https://api.gbif.org/v1/occurrence/search"
        params = {
            "taxonKey": nub_key,
            "mediaType": "StillImage",
            "limit": BATCH_SIZE,
            "offset": offset
        }
        
        try:
            response = requests.get(api_url, params=params).json()
            results = response.get("results", [])
            if not results: break # No more images for this species
            
            for record in results:
                if downloaded >= TOTAL_PER_SPECIES: break
                
                media = record.get("media", [])
                for item in media:
                    if item.get("type") == "StillImage":
                        img_url = item.get("identifier")
                        if not img_url: continue
                        
                        # Determine if it goes to train or val
                        target_dir = train_dir if downloaded < TRAIN_COUNT else val_dir
                        file_name = f"{taxon_key}_{downloaded}_{int(time.time())}.jpg"
                        save_path = os.path.join(target_dir, file_name)
                        
                        if download_image(img_url, save_path):
                            downloaded += 1
                            if downloaded % 50 == 0:
                                print(f"   {s_name}: Downloaded {downloaded}/{TOTAL_PER_SPECIES}")
                            break # Move to next record
            
            offset += BATCH_SIZE
            time.sleep(SLEEP_TIME) # Rate limiting
            
            if response.get("endOfRecords"): break

        except Exception as e:
            print(f"   Error fetching {s_name}: {e}")
            break

    print(f"[FINISH] {s_name}: Total {downloaded} images.")

def run_pipeline():
    print(f"Initializing Marine_v2 Dataset at {BASE_PATH}...")
    
    for family, genera in dataset_map.items():
        for genus, species_dict in genera.items():
            for species_name, taxon_key in species_dict.items():
                process_species(family, genus, species_name, taxon_key)


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
        nub_key = results[0].get('nubKey')
        canonical_name = results[0].get('canonicalName')
        print(f"Name: {canonical_name} - Nubkey: {nub_key}")
        return nub_key


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


async def process_species_async(f_name:str, g_name:str, s_name:str, wormsid:int):
    print(f"\n[START] Processing: {f_name} > {g_name} > {s_name} (ID: {wormsid})")
    # Create Directories
    train_dir = os.path.join(BASE_PATH, "train", f_name, g_name, s_name)
    val_dir = os.path.join(BASE_PATH, "val", f_name, g_name, s_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    total_assigned = 0  # Track how many we've decided to download
    downloaded_count = 0
    offset = 0
    semaphore = asyncio.Semaphore(20)  # Limit concurrent downloads
    connector = aiohttp.TCPConnector(limit=50)

    async with aiohttp.ClientSession(connector=connector) as session:
        nub_key = await get_nub_key_async(session,wormsid)
        if not nub_key: 
            return
        
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
                    print(f"   {s_name}: Progress {total_assigned}/{TOTAL_PER_SPECIES} assigned...")
                                   
                if data.get("endOfRecords"): 
                    break

                offset += BATCH_SIZE
                await asyncio.sleep(SLEEP_TIME)
                

            except Exception as e:
                print(f"   Error fetching {s_name}: {e}")
                break

    print(f"[FINISH] {s_name}: Total {downloaded_count} images.")


async def run_pipeline_async():
    print(f"Initializing Marine_v2 Dataset at {BASE_PATH}...")

    semaphore = asyncio.Semaphore(3)

    async def sem_tasks(family, genus, species_name, taxon_key):
        async with semaphore:
            await process_species_async(family, genus, species_name, taxon_key)
    
    tasks = []
    for family, genera in dataset_map.items():
        for genus, species_dict in genera.items():
            for species_name, taxon_key in species_dict.items():
                tasks.append(
                    asyncio.create_task(sem_tasks(family, genus, species_name, taxon_key))
                )
    await asyncio.gather(*tasks)

#---------------------------------------------------------------------------------------------------


if __name__ == "__main__":
    asyncio.run(run_pipeline_async())