from pathlib import Path
from CNN.Dataset.AudioVisualDataset import AudioVisualDataset
import time

print('Creating dataset...')
ds = AudioVisualDataset(Path('MFCCs_00'), Path('Visual_Features_00'))
print(f'Dataset size: {len(ds)}')

print('\nLoading all 800 samples...')
start = time.time()
shapes = []

for i in range(len(ds)):
    sample, label = ds[i]
    shapes.append(sample.shape)
    
    if (i + 1) % 100 == 0:
        elapsed = time.time() - start
        print(f'Loaded {i+1}/800 samples... ({elapsed:.1f}s elapsed)')

elapsed = time.time() - start
print(f'\n✅ Total time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)')
print(f'Average per sample: {elapsed/len(ds):.3f} seconds')
print(f'\nSample shapes (first 5): {shapes[:5]}')
print(f'Sample shapes (last 5): {shapes[-5:]}')
