"""
Data prep for vehicle viewpoint classifier.
Parses VIA annotations, infers viewpoint labels via voting heuristics,
and creates stratified train/val/test splits.
"""

import json
import os
from pathlib import Path
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

# CONFIGURATION

DATASET_PATH = Path("dataset")

# Part identity -> viewpoint vote mapping
# (ws=windscreen, orvm=side mirror, wa=wheel arch, qpanel=quarter panel)
FRONT_PARTS = {
    'frontheadlamp', 'frontbumper', 'frontbumpergrille',  
    'bonnet', 'frontws', 'towbarcover', 'lowerbumpergrille', 'frontbumpercladding'
}  # NOTE: 'logo' excluded - appears on front AND rear, caused labeling issues

REAR_PARTS = {
    'taillamp', 'tailgate', 'rearbumper', 'rearws',
    'lefttaillamp', 'righttaillamp', 'antenna', 'rearbumpercladding'
}

LEFT_PARTS = {
    'leftheadlamp', 'leftfoglamp', 'leftfrontdoor', 'leftreardoor',
    'leftwa', 'leftrunningboard', 'leftqpanel',
    'leftfrontdoorcladding', 'leftreardoorcladding', 
    'leftorvm', 'leftfender', 'leftapillar', 'lefttaillamp'
}

RIGHT_PARTS = {
    'rightheadlamp', 'rightfoglamp', 'rightfrontdoor', 'rightreardoor',
    'rightwa', 'rightrunningboard', 'rightqpanel',
    'rightorvm', 'rightfender', 'rightapillar', 'righttaillamp'
}

# Damage annotations - ignore for viewpoint detection
DAMAGE_PARTS = {
    'scratch', 'dent', 'dirt', 'd2', 'bumpertorn', 'bumpertear', 
    'bumperdent', 'crack', 'clipsbroken', 'rust'
}

CLASSES = ['Front', 'FrontLeft', 'FrontRight', 'Rear', 'RearLeft', 'RearRight', 'Background']


# LABEL EXTRACTION FUNCTIONS

def normalize_identity(identity: str) -> str:
    """Lowercase and strip 'partial_' prefix."""
    identity = identity.lower().strip()
    return identity[8:] if identity.startswith('partial_') else identity


def extract_viewpoint_label(regions: list) -> str:
    """Voting heuristic: count front/rear/left/right parts to determine viewpoint."""
    if not regions:
        return 'Background'
    
    # Collect all part identities
    identities = set()
    for region in regions:
        identity = region.get('region_attributes', {}).get('identity', '')
        if identity:
            identities.add(normalize_identity(identity))
    
    # Filter out damage-only annotations
    non_damage = identities - DAMAGE_PARTS
    if len(non_damage) < 2:
        return 'Background'
    
    # Count votes for each direction
    front_votes = len(identities & FRONT_PARTS)
    rear_votes = len(identities & REAR_PARTS)
    left_votes = len(identities & LEFT_PARTS)
    right_votes = len(identities & RIGHT_PARTS)
    
    # Determine primary axis (front vs rear)
    if front_votes > rear_votes:
        primary = 'Front'
    elif rear_votes > front_votes:
        primary = 'Rear'
    elif front_votes > 0:
        primary = 'Front'  # Tie-break: prefer Front
    else:
        # No clear front/rear indicators
        # Check if we have strong left/right indicators
        if left_votes > 0 or right_votes > 0:
            # Default to Front for side views without clear front/rear
            primary = 'Front'
        else:
            return 'Background'
    
    # Determine secondary axis (left vs right)
    if left_votes > right_votes:
        secondary = 'Left'
    elif right_votes > left_votes:
        secondary = 'Right'
    else:
        secondary = ''  # Pure front/rear view
    
    return primary + secondary


def parse_dataset(dataset_path: Path) -> pd.DataFrame:
    """Walk dataset folders, parse VIA JSON, return DataFrame of (filepath, filename, label)."""
    data = []
    
    for folder in sorted(dataset_path.iterdir()):
        if not folder.is_dir():
            continue
        
        json_path = folder / "via_region_data.json"
        if not json_path.exists():
            continue
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: Could not parse {json_path}: {e}")
            continue
        
        for key, entry in annotations.items():
            filename = entry.get('filename', '')
            if not filename:
                continue
            
            # Check if image file exists
            image_path = folder / filename
            if not image_path.exists():
                # Try alternate extensions
                for ext in ['.jpg', '.jpeg', '.png']:
                    alt_path = folder / (Path(filename).stem + ext)
                    if alt_path.exists():
                        image_path = alt_path
                        filename = alt_path.name
                        break
            
            if not image_path.exists():
                continue
            
            regions = entry.get('regions', [])
            label = extract_viewpoint_label(regions)
            
            data.append({
                'filepath': str(image_path),
                'filename': filename,
                'label': label
            })
    
    return pd.DataFrame(data)


def create_stratified_splits(df: pd.DataFrame, 
                             train_ratio: float = 0.8,
                             val_ratio: float = 0.1,
                             test_ratio: float = 0.1,
                             random_state: int = 42) -> tuple:
    """80/10/10 stratified split."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio,
        stratify=df['label'],
        random_state=random_state
    )
    
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_ratio,
        stratify=temp_df['label'],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def print_distribution(df: pd.DataFrame, name: str):
    print(f"\n{name} ({len(df)} samples):")
    counts = df['label'].value_counts().sort_index()
    for label, count in counts.items():
        print(f"  {label:12s}: {count:4d} ({100*count/len(df):5.1f}%)")


# MAIN

def main():
    print("=" * 60)
    print("Vehicle Viewpoint Classifier - Data Preparation")
    print("=" * 60)
    
    print(f"\nParsing dataset from: {DATASET_PATH.absolute()}")
    df = parse_dataset(DATASET_PATH)
    print(f"Total images: {len(df)}")
    
    print_distribution(df, "Overall")
    
    missing = set(CLASSES) - set(df['label'].unique())
    if missing:
        print(f"\nWarning: Missing classes: {missing}")
    
    print("\nCreating stratified splits (80/10/10)...")
    train_df, val_df, test_df = create_stratified_splits(df)
    
    print_distribution(train_df, "Train")
    print_distribution(val_df, "Val")
    print_distribution(test_df, "Test")
    
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    print(f"\nSaved: train.csv ({len(train_df)}), val.csv ({len(val_df)}), test.csv ({len(test_df)})")
    
    labels_path = Path('models/saved_model')
    labels_path.mkdir(parents=True, exist_ok=True)
    with open(labels_path / 'labels.txt', 'w') as f:
        for cls in CLASSES:
            f.write(f"{cls}\n")
    print(f"Labels: {labels_path / 'labels.txt'}")


if __name__ == '__main__':
    main()
