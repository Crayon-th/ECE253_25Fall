import os
import shutil

# Create directory
test_dir = os.path.join('data', 'single_test')
os.makedirs(test_dir, exist_ok=True)
print(f'Created directory: {test_dir}')

# Copy the image
src = os.path.join('..', 'outputs', 'quick_demo', '4_rand_noisy.png')
dst = os.path.join('data', 'single_test', '4_rand_noisy.png')

if os.path.exists(src):
    shutil.copy(src, dst)
    print(f'Copied file to: {dst}')
    print(f'File exists: {os.path.exists(dst)}')
else:
    print(f'Source file not found: {src}')
    print('Please check the path to your image file.')

