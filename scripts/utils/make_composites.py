import argparse
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm # Add progress bar

def pad_smaller_side(image):
  # Get the dimensions of the before image
  _height, _width, _ = image.shape

  # Calculate the amount of black padding needed to make the aspect ratio square
  if _height > _width:
    padding_height = 0
    padding_width = (_height - _width) // 2
  elif _height < _width:
    padding_height = (_width - _height) // 2
    padding_width = 0
  else:
    padding_height = 0
    padding_width = 0

  # Add black padding to the before image
  return cv2.copyMakeBorder(
    image,
    padding_height,
    padding_height,
    padding_width,
    padding_width,
    cv2.BORDER_CONSTANT,
    value=(0, 0, 0)
  )


def make_composite(before_file, after_file, save_dir):
  # the componenets have been tested so use try except
  # to just skip an image if any error occurs unexpectedly
  try:
    # Read in images
    images = [cv2.cvtColor(cv2.imread(before_file), cv2.COLOR_BGR2RGB),
              cv2.cvtColor(cv2.imread(after_file), cv2.COLOR_BGR2RGB)]
    # add padding to both
    images = [pad_smaller_side(x) for x in images]
    # resize both
    images = [tf.image.resize(x, [256, 256]) for x in images]
    
    # Combine (before on left after on right)
    combined_image = tf.concat([
      tf.dtypes.cast(images[0], tf.uint8),
      tf.dtypes.cast(images[1], tf.uint8)
    ], axis=1)
    combined_image = tf.io.encode_jpeg(combined_image, quality=100, format='rgb')
    
    # Save the combined image to the test_data/train directory
    filename = before_file.split('/')[-1]
    destination = os.path.join(save_dir, filename)
    tf.io.write_file(destination, combined_image)
  except:
    pass


def main():
  parser = argparse.ArgumentParser(description="With CSV containing paths to the image components, read in the file pairs and create a composite image.")
  parser.add_argument("--csv_file", required=True, help="Path to the CSV file.")
  parser.add_argument("--before_col", required=True, help="Name of the 'before' column.")
  parser.add_argument("--after_col", required=True, help="Name of the 'after' column.")
  parser.add_argument("--save_dir", required=True, help="Where to save composite images.")

  args = parser.parse_args()

  df = pd.read_csv(args.csv_file, index_col = None)
  for i, row in tqdm(df.iterrows(), total=len(df)):
    make_composite(row[args.before_col], row[args.after_col], args.save_dir)

if __name__ == "__main__":
  main()

