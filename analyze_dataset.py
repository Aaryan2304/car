"""Quick dataset analysis script to understand image and annotation structure."""
import json
import os
from pathlib import Path
from collections import Counter

DATASET_PATH = Path(r"c:\Users\aarya\Music\car\dataset")

# Collect all part identities across dataset
all_identities = Counter()
image_count = 0
folder_stats = []

# Track viewpoint-related parts specifically
FRONT_PARTS = {'frontheadlamp', 'frontbumper', 'frontbumpergrille', 'logo', 
               'bonnet', 'frontws', 'towbarcover', 'lowerbumpergrille', 'frontbumpercladding'}
REAR_PARTS = {'taillamp', 'tailgate', 'rearbumper', 'rearws', 'lefttaillamp', 
              'righttaillamp', 'antenna', 'rearbumpercladding'}
LEFT_PARTS = {'leftheadlamp', 'leftfoglamp', 'leftfrontdoor', 'leftreardoor',
              'leftwa', 'leftrunningboard', 'leftqpanel', 'leftfrontdoorcladding',
              'leftreardoorcladding', 'leftorvm', 'leftfender', 'leftapillar', 'lefttaillamp'}
RIGHT_PARTS = {'rightheadlamp', 'rightfoglamp', 'rightfrontdoor', 'rightreardoor',
               'rightwa', 'rightrunningboard', 'rightqpanel', 'rightorvm', 'rightfender', 
               'rightapillar', 'righttaillamp'}

viewpoint_counts = Counter()
empty_annotations = 0
damage_only_images = 0

DAMAGE_PARTS = {'scratch', 'dent', 'dirt', 'd2', 'bumpertorn', 'bumpertear', 'bumperdent', 
                'crack', 'clipsbroken', 'rust'}

for folder in DATASET_PATH.iterdir():
    if not folder.is_dir():
        continue
    
    json_path = folder / "via_region_data.json"
    if not json_path.exists():
        continue
    
    # Count images in folder
    images = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg"))
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        continue
    
    folder_identities = Counter()
    for entry in data.values():
        regions = entry.get('regions', [])
        
        if len(regions) == 0:
            empty_annotations += 1
            continue
            
        identities = set()
        for region in regions:
            identity = region.get('region_attributes', {}).get('identity', '').lower()
            # Handle partial_ prefix
            if identity.startswith('partial_'):
                identity = identity[8:]
            if identity:
                identities.add(identity)
                all_identities[identity] += 1
                folder_identities[identity] += 1
        
        # Check if damage-only
        non_damage = identities - DAMAGE_PARTS
        if len(non_damage) == 0 and len(identities) > 0:
            damage_only_images += 1
            continue
        
        # Count viewpoint votes
        front = len(identities & FRONT_PARTS)
        rear = len(identities & REAR_PARTS)
        left = len(identities & LEFT_PARTS)
        right = len(identities & RIGHT_PARTS)
        
        # Determine primary direction
        if front > rear:
            primary = 'Front'
        elif rear > front:
            primary = 'Rear'
        elif front > 0:
            primary = 'Front'
        else:
            primary = None
            
        # Determine side
        if left > right:
            secondary = 'Left'
        elif right > left:
            secondary = 'Right'
        else:
            secondary = ''
            
        if primary:
            label = primary + secondary
            viewpoint_counts[label] += 1
        else:
            viewpoint_counts['Background'] += 1
    
    image_count += len(images)
    folder_stats.append({
        'folder': folder.name,
        'images': len(images),
        'annotations': len(data),
        'top_parts': folder_identities.most_common(5)
    })

print(f"\n=== DATASET OVERVIEW ===")
print(f"Total folders: {len(folder_stats)}")
print(f"Total images: {image_count}")

print(f"\n=== VIEWPOINT DISTRIBUTION (from heuristics) ===")
for label, count in viewpoint_counts.most_common():
    print(f"  {label}: {count}")
    
print(f"\n=== POTENTIAL BACKGROUND SOURCES ===")
print(f"  Empty annotations: {empty_annotations}")
print(f"  Damage-only images: {damage_only_images}")

print(f"\n=== ALL PART IDENTITIES (sorted by frequency, top 40) ===")
for identity, count in all_identities.most_common(40):
    print(f"  {identity}: {count}")

print(f"\n=== SAMPLE IMAGE ANNOTATIONS ===")
sample_folder = DATASET_PATH / "5e9112c35026365e15eb871b"
sample_json = sample_folder / "via_region_data.json"
with open(sample_json, 'r') as f:
    data = json.load(f)

for i, (key, entry) in enumerate(data.items()):
    if i >= 5:
        break
    filename = entry.get('filename', 'N/A')
    regions = entry.get('regions', [])
    identities = [r.get('region_attributes', {}).get('identity', '') for r in regions]
    print(f"\n{filename}:")
    print(f"  Parts ({len(identities)}): {identities}")
