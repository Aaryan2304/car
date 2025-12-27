# Data Preparation Script - Deep Dive

## File: `data_preparation.py`

This document explains every aspect of the data preparation script for your interview.

---

## 1. The Problem: No Direct Labels

### What I Was Given
The dataset contains 61 folders, each with:
- Multiple vehicle images (`.jpg`, `.jpeg`)
- A `via_region_data.json` file with polygon annotations

### The Challenge
**There are no explicit viewpoint labels.** The JSON files contain part-level annotations like `frontbumper`, `leftheadlamp`, `taillamp` - but not "this is a FrontLeft image."

### My Solution: Voting Heuristics
I designed a deterministic algorithm that **infers viewpoint from visible parts**. The logic:
- If you can see the `frontbumper` and `bonnet`, you're looking at the front
- If you also see `leftfrontdoor` and `leftorvm`, you're looking from the front-left angle
- Note: `logo` is NOT used for voting since logos appear on both front AND rear of vehicles

---

## 2. Part-to-Viewpoint Mapping

### Why These Specific Parts?

I analyzed the dataset using `analyze_dataset.py` to find all unique part identities. Then I categorized them:

```python
FRONT_PARTS = {
    'frontheadlamp', 'frontbumper', 'frontbumpergrille',  # 'logo' excluded - see note below
    'bonnet', 'frontws', 'towbarcover', 'lowerbumpergrille', 'frontbumpercladding'
}
```

**Interview Q: Why is `logo` NOT included in FRONT_PARTS?**
> Initially I considered including `logo` as a front indicator since many vehicles have a prominent front grille logo. However, **logos also appear on the rear** (trunk/tailgate badge). Including `logo` would incorrectly vote for "Front" even when viewing from behind. It's ambiguous, so I excluded it.

**Interview Q: Why is `bonnet` (hood) a FRONT part?**
> The hood/bonnet is only visible from the front. If you're behind the car, you can't see it. It's unambiguous.

**Interview Q: What is `frontws`?**
> WS = Windscreen (windshield). The front windscreen is only visible when viewing from the front.

```python
REAR_PARTS = {
    'taillamp', 'tailgate', 'rearbumper', 'rearws', 'lefttaillamp', 
    'righttaillamp', 'antenna', 'rearbumpercladding'
}
```

**Interview Q: Why are `lefttaillamp` and `righttaillamp` in REAR_PARTS, not LEFT/RIGHT?**
> Great question! Even though they have "left/right" in the name, taillamps are only visible when you're looking at the rear of the vehicle. They tell us REAR primarily. I also added them to LEFT_PARTS/RIGHT_PARTS respectively for the secondary axis.

```python
LEFT_PARTS = {
    'leftheadlamp', 'leftfoglamp', 'leftfrontdoor', 'leftreardoor',
    'leftwa', 'leftrunningboard', 'leftqpanel', 'leftfrontdoorcladding',
    'leftreardoorcladding', 'leftorvm', 'leftfender', 'leftapillar', 'lefttaillamp'
}
```

**Interview Q: What is `leftwa`?**
> WA stands for "wheel arch" - the curved body panel above the wheel. It's a left-side indicator.

**Interview Q: What is `leftorvm`?**
> ORVM = Outside Rear View Mirror (side mirror). If you see the left mirror, you're viewing from the left side.

### Damage Parts - Why Ignore Them?

```python
DAMAGE_PARTS = {
    'scratch', 'dent', 'dirt', 'd2', 'bumpertorn', 'bumpertear', 
    'bumperdent', 'crack', 'clipsbroken', 'rust'
}
```

**Interview Q: Why ignore damage parts for viewpoint detection?**
> Damage annotations tell us about the vehicle's condition, not which angle we're viewing from. A `scratch` could appear on any panel from any angle. Including them would add noise to the viewpoint voting.

### Automotive Abbreviation Glossary

**Interview Q: How did you know what `leftwa`, `leftorvm`, `rearws` etc. mean?**
> These are standard automotive industry abbreviations. I inferred their meanings through:
> 1. **Context**: Looking at which images these parts appeared in
> 2. **Domain knowledge**: Standard automotive terminology
> 3. **Pattern matching**: Seeing related terms together (e.g., `leftorvm` appears with other left-side parts)
>
> In a real project, I would ask the data annotation team for a glossary to confirm.

| Abbreviation | Full Form | Description |
|--------------|-----------|-------------|
| `ws` | Windscreen | Front or rear windshield/windscreen |
| `orvm` | Outside Rear View Mirror | Side mirror (driver/passenger side) |
| `wa` | Wheel Arch | Curved body panel above each wheel |
| `qpanel` | Quarter Panel | Rear side body panel behind doors |
| `apillar` | A-Pillar | Front windshield support pillar |

**Interview Q: What if some abbreviations were unclear?**
> In production, I would:
> 1. Request documentation from the annotation team
> 2. Sample images with unknown parts and visually identify them
> 3. If still unclear, treat them as "neutral" (don't add to any viewpoint set)

---

## 3. The Voting Algorithm

### `normalize_identity()` Function

```python
def normalize_identity(identity: str) -> str:
    identity = identity.lower().strip()
    if identity.startswith('partial_'):
        identity = identity[8:]
    return identity
```

**Purpose**: Standardize part names before matching.

**Interview Q: Why strip 'partial_' prefix?**
> Some annotations are marked as `partial_frontbumper` when only part of the bumper is visible. For viewpoint detection, a partial view of the front bumper still indicates we're looking at the front. So I treat `partial_frontbumper` the same as `frontbumper`.

### `extract_viewpoint_label()` Function - The Core Logic

```python
def extract_viewpoint_label(regions: list) -> str:
    if not regions:
        return 'Background'
```

**First check**: Empty annotations = Background. If there are no polygon regions, it's either a non-car image or too zoomed in.

```python
    # Collect all part identities
    identities = set()
    for region in regions:
        identity = region.get('region_attributes', {}).get('identity', '')
        if identity:
            identities.add(normalize_identity(identity))
```

**Why use a set?** To avoid counting duplicate parts. If `frontbumper` appears twice in the annotations (maybe split polygons), we should count it once.

```python
    # Filter out damage-only annotations
    non_damage = identities - DAMAGE_PARTS
    if len(non_damage) < 2:
        return 'Background'
```

**Interview Q: Why require at least 2 non-damage parts?**
> With fewer than 2 meaningful parts, we don't have enough information to determine viewpoint. A single `tyre` annotation could be any angle. Requiring 2+ parts gives us more confidence. This also catches images that only show damage (scratches, dents) without clear vehicle structure.

```python
    # Count votes for each direction
    front_votes = len(identities & FRONT_PARTS)
    rear_votes = len(identities & REAR_PARTS)
    left_votes = len(identities & LEFT_PARTS)
    right_votes = len(identities & RIGHT_PARTS)
```

**Set intersection** (`&`) efficiently counts how many parts belong to each category.

### Primary Axis Resolution (Front vs Rear)

```python
    if front_votes > rear_votes:
        primary = 'Front'
    elif rear_votes > front_votes:
        primary = 'Rear'
    elif front_votes > 0:
        primary = 'Front'  # Tie-break: prefer Front
    else:
        if left_votes > 0 or right_votes > 0:
            primary = 'Front'
        else:
            return 'Background'
```

**Interview Q: Why prefer Front in a tie?**
> Two reasons:
> 1. **Data distribution**: Front views are slightly more common in the dataset
> 2. **Business logic**: In vehicle inspection, front-angle photos are typically taken first and are more common

**Interview Q: What if there are only LEFT/RIGHT parts visible?**
> This is an edge case - a pure side view. I default to `Front` as the primary because true 90-degree side views are rare. Usually there's some front or rear visible. This could be refined with more data analysis.

### Secondary Axis Resolution (Left vs Right)

```python
    if left_votes > right_votes:
        secondary = 'Left'
    elif right_votes > left_votes:
        secondary = 'Right'
    else:
        secondary = ''  # Pure front/rear view
```

**Interview Q: What does empty string mean?**
> It means this is a pure front or rear view with no left/right bias. The final label will be just "Front" or "Rear" without a direction suffix.

```python
    return primary + secondary  # e.g., "Front" + "Left" = "FrontLeft"
```

---

## 4. Parsing VIA JSON Format

### Understanding the JSON Structure

The VIA (VGG Image Annotator) format looks like:
```json
{
  "image1.jpg12345": {
    "filename": "image1.jpg",
    "size": 12345,
    "regions": [
      {
        "shape_attributes": {"name": "polygon", "all_points_x": [...], "all_points_y": [...]},
        "region_attributes": {"identity": "frontbumper"}
      }
    ]
  }
}
```

**Interview Q: Why not use the JSON key directly as filename?**
> The key is `filename + size` concatenated, like `image1.jpg12345`. I use the `filename` field inside the entry for the actual filename because it's reliable.

### The `parse_dataset()` Function

```python
def parse_dataset(dataset_path: Path) -> pd.DataFrame:
    data = []
    
    for folder in sorted(dataset_path.iterdir()):
        if not folder.is_dir():
            continue
```

**Why sorted?** Reproducibility. Different systems might iterate folders in different orders. Sorting ensures consistent output.

```python
        json_path = folder / "via_region_data.json"
        if not json_path.exists():
            continue
```

Some folders might be empty or missing JSON files. Skip gracefully.

```python
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"Warning: Could not parse {json_path}: {e}")
            continue
```

**Interview Q: Why catch both exceptions?**
> `JSONDecodeError`: Malformed JSON syntax
> `UnicodeDecodeError`: Encoding issues (some files might not be UTF-8)
> I print a warning but continue processing other folders - don't let one bad file crash the whole script.

```python
        for key, entry in annotations.items():
            filename = entry.get('filename', '')
            if not filename:
                continue
            
            image_path = folder / filename
            if not image_path.exists():
                # Try alternate extensions
                for ext in ['.jpg', '.jpeg', '.png']:
                    alt_path = folder / (Path(filename).stem + ext)
                    if alt_path.exists():
                        image_path = alt_path
                        filename = alt_path.name
                        break
```

**Interview Q: Why try alternate extensions?**
> Sometimes the JSON references `image.jpg` but the actual file is `image.jpeg` or has different casing. This fallback increases data recovery.

---

## 5. The Background Class - Detailed Explanation

### What is the Background Class?

**IMPORTANT**: The `Background` class is NOT about specific car parts. It's a **catch-all category for images where the viewpoint cannot be reliably determined.**

### When is an Image Labeled as Background?

The `extract_viewpoint_label()` function returns `'Background'` in these situations:

**1. Empty Annotations** (lines 70-71 in code):
```python
if not regions:
    return 'Background'
```
- Image has `regions: []` in the JSON
- This could be: non-car images, heavily cropped images, or unannotated images

**2. Damage-Only Annotations** (lines 82-84 in code):
```python
non_damage = identities - DAMAGE_PARTS
if len(non_damage) < 2:
    return 'Background'
```
- Image only shows scratches, dents, rust, etc.
- Example: Close-up of a scratch with no identifiable car structure
- With <2 meaningful parts, we can't determine viewing angle

**3. Ambiguous/Insufficient Parts** (lines 97-102 in code):
```python
else:
    if left_votes > 0 or right_votes > 0:
        primary = 'Front'  # Default assumption
    else:
        return 'Background'
```
- Only neutral parts visible (like `tyre` which appears on all sides)
- No front/rear/left/right indicators

### Examples of Background Images

| Scenario | Why Background? |
|----------|----------------|
| Empty parking lot | No car visible (`regions: []`) |
| Close-up of scratch on door | Only `scratch` annotation, no structure |
| Single tyre visible | Only 1 neutral part, can't determine angle |
| Blurry/unusable image | Annotator left it empty |
| Interior shot | Interior parts not mapped to viewpoints |

### Interview Q: Why not have a separate "Unknown" class?
> `Background` effectively IS the "Unknown" class. I called it Background because:
> 1. Many cases are literally background/non-vehicle images
> 2. The assignment spec mentioned "Background/Unknown" as one category
> 3. For the model, it learns to recognize "this isn't a clear vehicle viewpoint"

### Interview Q: How many Background samples are there?
> Approximately 472 images (~12% of dataset). This includes:
> - 57 images with completely empty annotations
> - 21 images with only damage annotations
> - ~394 images with insufficient/ambiguous parts

### Interview Q: Is Background the hardest class to classify?
> Yes! It has the lowest F1 score (0.650) because:
> 1. **Heterogeneous**: Diverse appearances (empty lots, close-ups, partial views)
> 2. **Limited samples**: Only 472 training samples
> 3. **Negative definition**: Defined by what it's NOT, not what it IS

---

## 6. Stratified Splitting

### Why 80/10/10 Instead of 70/20/10?

```python
def create_stratified_splits(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_state=42):
```

**Interview Q: Why not use 70/20/10 which is more standard?**

Both splits are valid. Here's the trade-off:

| Split | Train | Val | Test | Analysis |
|-------|-------|-----|------|----------|
| 80/10/10 | 3,179 | 397 | 398 | More training data, smaller validation |
| 70/20/10 | 2,780 | 795 | 398 | Less training data, larger validation |

**Why I chose 80/10/10**:
1. **Small dataset**: With only ~4000 images, every training sample matters
2. **Transfer learning**: Using pretrained MobileNetV2, we need less data to fine-tune
3. **Class imbalance**: Some classes have <350 samples; 70% would give <245 training samples for minority classes
4. **Validation purpose**: 397 samples is sufficient to detect overfitting trends

**When 70/20/10 would be better**:
1. **Larger datasets** (>50K images): More validation data helps tune hyperparameters
2. **No transfer learning**: Training from scratch needs more validation signal
3. **Extensive hyperparameter search**: More validation data = more reliable comparison

**Interview Q: How would you decide between splits?**
> I'd consider:
> - Dataset size (smaller → prioritize training)
> - Transfer learning (yes → can use less data)
> - Hyperparameter search scope (extensive → need more validation)
> - Minority class sizes (ensure enough samples per class in all splits)

### Why Stratified?

**Interview Q: What is stratified splitting?**
> Stratified splitting ensures each split (train/val/test) has the same class distribution as the original dataset. Without stratification, you might randomly get a test set with no "Rear" images, making evaluation unfair.

### The Two-Step Split

```python
    train_df, temp_df = train_test_split(
        df, 
        train_size=train_ratio,
        stratify=df['label'],
        random_state=random_state
    )
```

First split: 80% train, 20% temp (val + test combined)

```python
    relative_val_ratio = val_ratio / (val_ratio + test_ratio)  # = 0.1 / 0.2 = 0.5
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_ratio,
        stratify=temp_df['label'],
        random_state=random_state
    )
```

**Interview Q: Why calculate relative_val_ratio?**
> After the first split, we have 20% of data left. We want 10% for val and 10% for test, which is a 50/50 split of the remaining data. So `relative_val_ratio = 0.5`.

**Interview Q: Why use random_state=42?**
> Reproducibility. Anyone running this script will get the same train/val/test splits. The value 42 is conventional (Hitchhiker's Guide reference), but any fixed integer works.

---

## 7. Output Files

The script produces:
- `train.csv`, `val.csv`, `test.csv` with columns: `filepath`, `filename`, `label`
- `models/saved_model/labels.txt` with the 7 class names

**Interview Q: Why save labels.txt separately?**
> The model needs to know the class order at inference time. labels.txt is a simple, portable format that works with both SavedModel and TFLite inference.

---

## 8. Potential Interview Questions

**Q: How would you handle a new part identity not in your mappings?**
> It would be ignored for voting purposes. I could add a warning log and periodically review unmapped parts to update the mappings.

**Q: What if the voting produces ambiguous results frequently?**
> I'd analyze the edge cases, potentially add centroid-based tie-breaking (comparing average X-position of left vs right parts), or flag ambiguous cases for manual review.

**Q: Why not use machine learning for label extraction?**
> With only ~4000 images and no ground truth labels, training a separate ML model for labeling would be risky. The heuristic approach is transparent, debuggable, and uses domain knowledge directly.

**Q: How would you validate the heuristic accuracy?**
> Manual spot-checking. Sample 50-100 images, visually verify the assigned label, and calculate accuracy. If below 90%, refine the heuristics.
