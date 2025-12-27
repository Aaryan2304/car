"""
Data Preparation Script for ClearQuote Vehicle Viewpoint Classifier

This script:
1. Parses VIA JSON annotation files from all dataset folders
2. Infers viewpoint labels using deterministic voting heuristics
3. Detects Background class from empty/sparse annotations
4. Creates stratified train/val/test splits (80/10/10)
5. Outputs: train.csv, val.csv, test.csv
"""

import json
import os
from pathlib import Path
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_PATH = Path("dataset")

# Part identity to viewpoint vote mapping (from dataset analysis)
# Abbreviations: ws=windscreen, orvm=outside rear view mirror, wa=wheel arch, qpanel=quarter panel
FRONT_PARTS = {
    'frontheadlamp', 'frontbumper', 'frontbumpergrille',  
    'bonnet', 'frontws', 'towbarcover', 'lowerbumpergrille', 'frontbumpercladding'
}

REAR_PARTS = {
    'taillamp', 'tailgate', 'rearbumper', 'rearws',  # rearws = rear windscreen
    'lefttaillamp', 'righttaillamp', 'antenna', 'rearbumpercladding'
}

LEFT_PARTS = {
    'leftheadlamp', 'leftfoglamp', 'leftfrontdoor', 'leftreardoor',
    'leftwa',  # wa = wheel arch (curved panel above wheel)
    'leftrunningboard', 'leftqpanel',  # qpanel = quarter panel (rear side body)
    'leftfrontdoorcladding', 'leftreardoorcladding', 
    'leftorvm',  # orvm = outside rear view mirror (side mirror)
    'leftfender', 'leftapillar', 'lefttaillamp'
}

RIGHT_PARTS = {
    'rightheadlamp', 'rightfoglamp', 'rightfrontdoor', 'rightreardoor',
    'rightwa', 'rightrunningboard', 'rightqpanel',  # Same abbreviations as LEFT_PARTS
    'rightorvm', 'rightfender', 'rightapillar', 'righttaillamp'
}

# Damage/irrelevant parts to ignore for viewpoint detection
DAMAGE_PARTS = {
    'scratch', 'dent', 'dirt', 'd2', 'bumpertorn', 'bumpertear', 
    'bumperdent', 'crack', 'clipsbroken', 'rust'
}

# All 7 classes
CLASSES = ['Front', 'FrontLeft', 'FrontRight', 'Rear', 'RearLeft', 'RearRight', 'Background']


# =============================================================================
# LABEL EXTRACTION FUNCTIONS
# =============================================================================

def normalize_identity(identity: str) -> str:
    """Normalize part identity: lowercase and strip 'partial_' prefix."""
    identity = identity.lower().strip()
    if identity.startswith('partial_'):
        identity = identity[8:]
    return identity


def extract_viewpoint_label(regions: list) -> str:
    """
    Extract viewpoint label from image regions using voting heuristics.
    
    Returns one of: Front, FrontLeft, FrontRight, Rear, RearLeft, RearRight, Background
    """
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
    """
    Parse all VIA JSON files and extract image paths with inferred labels.
    
    Returns DataFrame with columns: filepath, filename, label
    """
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
    """
    Create stratified train/val/test splits.
    
    Returns: (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio,
        stratify=df['label'],
        random_state=random_state
    )
    
    # Second split: val vs test (from remaining data)
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_ratio,
        stratify=temp_df['label'],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def print_distribution(df: pd.DataFrame, name: str):
    """Print label distribution for a dataset split."""
    print(f"\n{name} Distribution ({len(df)} samples):")
    counts = df['label'].value_counts().sort_index()
    for label, count in counts.items():
        pct = 100 * count / len(df)
        print(f"  {label:12s}: {count:4d} ({pct:5.1f}%)")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("ClearQuote Vehicle Viewpoint Classifier - Data Preparation")
    print("=" * 60)
    
    # Parse dataset
    print(f"\nParsing dataset from: {DATASET_PATH.absolute()}")
    df = parse_dataset(DATASET_PATH)
    print(f"Total images found: {len(df)}")
    
    # Print overall distribution
    print_distribution(df, "Overall Dataset")
    
    # Verify all classes are present
    missing_classes = set(CLASSES) - set(df['label'].unique())
    if missing_classes:
        print(f"\nWarning: Missing classes: {missing_classes}")
    
    # Create stratified splits
    print("\nCreating stratified splits (80/10/10)...")
    train_df, val_df, test_df = create_stratified_splits(df)
    
    # Print split distributions
    print_distribution(train_df, "Training Set")
    print_distribution(val_df, "Validation Set")
    print_distribution(test_df, "Test Set")
    
    # Save to CSV
    train_df.to_csv('train.csv', index=False)
    val_df.to_csv('val.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    
    print("\n" + "=" * 60)
    print("Output files created:")
    print(f"  - train.csv ({len(train_df)} samples)")
    print(f"  - val.csv ({len(val_df)} samples)")
    print(f"  - test.csv ({len(test_df)} samples)")
    print("=" * 60)
    
    # Save class labels
    labels_path = Path('models/saved_model')
    labels_path.mkdir(parents=True, exist_ok=True)
    with open(labels_path / 'labels.txt', 'w') as f:
        for cls in CLASSES:
            f.write(f"{cls}\n")
    print(f"\nLabels saved to: {labels_path / 'labels.txt'}")


if __name__ == '__main__':
    main()
