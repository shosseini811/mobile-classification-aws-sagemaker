import os
import shutil
import numpy as np
import boto3
from sklearn.model_selection import train_test_split

s3 = boto3.client('s3')
bucket_name = 'sourceofimages'

# Delete all objects in the bucket
s3_bucket = boto3.resource('s3').Bucket(bucket_name)
s3_bucket.objects.all().delete()

categories = ['good', 'oil', 'scratch', 'stain']
data_dir = '.'  # Current directory

# Create train and validation directories
os.makedirs('train', exist_ok=True)
os.makedirs('validation', exist_ok=True)

for category in categories:
    # Create category directories inside train and validation directories
    os.makedirs(os.path.join('train', category), exist_ok=True)
    os.makedirs(os.path.join('validation', category), exist_ok=True)

    # Get list of all image files for this category
    image_files = os.listdir(os.path.join(data_dir, category))
    image_files = [f for f in image_files if f.endswith('.jpg') or f.endswith('.png')]

    # Split into training and validation sets (80% train, 20% validation)
    train_files, val_files = train_test_split(image_files, test_size=0.2)

    # Move training files to train directory
    for file in train_files:
        shutil.move(os.path.join(data_dir, category, file), os.path.join('train', category, file))

    # Move validation files to validation directory
    for file in val_files:
        shutil.move(os.path.join(data_dir, category, file), os.path.join('validation', category, file))

# Upload data to S3
for dataset_type in ['train', 'validation']:
    for category in categories:
        for file in os.listdir(os.path.join(dataset_type, category)):
            s3.upload_file(os.path.join(dataset_type, category, file), bucket_name, os.path.join(dataset_type, category, file))
