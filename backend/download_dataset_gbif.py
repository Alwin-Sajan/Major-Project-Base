import os
import requests
import time
import shutil
from concurrent.futures import ThreadPoolExecutor

# --- CONFIGURATION ---
BASE_PATH = "/media/abk/New Disk/DATASETS/marine_v2"
TRAIN_COUNT = 500
VAL_COUNT = 50
TOTAL_PER_SPECIES = TRAIN_COUNT + VAL_COUNT
BATCH_SIZE = 100 # GBIF API limit per request
SLEEP_TIME = 1.5  # Seconds between API calls to avoid rate limits

# --- TAXONOMY DATA ---
# Structure: { Family: { Genus: { Species_Name: taxonKey } } }
dataset_map = {
    "Gobiidae": {
        "Amblyeleotris": {
            "aurora": 2378457, "steinitzi": 2378812, "guttata": 2378829, "wheeleri": 2378453,
            "latifasciata": 2378417, "periophthalma": 2378802, "diagonalis": 2378466, "ogurae": 2378416
        },
        "Gobiodon": {
            "okinawae": 5208802, "histrio": 2378619, "citrinus": 5208803, "rivulatus": 2378622,
            "quinquestrigatus": 2378606, "erythrospilus": 2378615, "ceramensis": 2378608, "axillaris": 2378617
        }
    },
    "Serranidae": {
        "Epinephelus": {
            "fasciatus": 2388344, "merra": 2388271, "coioides": 2388213, "striatus": 2388300,
            "guttatus": 2388558, "marginatus": 2388507, "polyphekadion": 2388423, "diacanthus": 2388126
        },
        "Cephalopholis": {
            "argus": 2388891, "miniata": 2381788, "sexmaculata": 2388884, "urodeta": 2388881,
            "sonnerati": 2388907, "cruentata": 2388903, "leoparda": 2388887, "taeniops": 2388910
        }
    }
}

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

def process_species(f_name, g_name, s_name, taxon_key):
    print(f"\n[START] Processing: {f_name} > {g_name} > {s_name} (ID: {taxon_key})")
    
    # Create Directories
    train_dir = os.path.join(BASE_PATH, "train", f_name, g_name, s_name)
    val_dir = os.path.join(BASE_PATH, "val", f_name, g_name, s_name)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    downloaded = 0
    offset = 0
    
    while downloaded < TOTAL_PER_SPECIES:
        api_url = "https://api.gbif.org/v1/occurrence/search"
        params = {
            "taxonKey": taxon_key,
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

if __name__ == "__main__":
    run_pipeline()