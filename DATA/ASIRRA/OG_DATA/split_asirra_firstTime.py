# This file explains how the original ASIRRA data
# is split into training/testing.
#FROM:
#  https://github.com/desimone/pytorch-cat-vs-dogs
#



# unzip!
  unzip train.zip
  unzip test.zip
  # prep train directory and split train/trainval
  mv train/ catdog
  cd catdog
  # sanity check
  find . -type f -name 'cat*' | wc -l # 12500
  find . -type f -name 'dog*' | wc -l # 12500
  mkdir -p train/dog
  mkdir -p train/cat
  mkdir -p test/dog   #test instead of "val"
  mkdir -p test/cat   #test instead of "val"
  # Randomly move 90% into train and test, 
  # if reproducability is important you can pass in a source to shuf
#  no clue how 
  find . -name "cat*" -type f | shuf -n11250 | xargs -I file mv file train/cat/
  find . -maxdepth 1 -type f -name 'cat*'| xargs -I file mv file test/cat/
  # now dogs
  find . -name "dog*" -type f | shuf -n11250 | xargs -I file mv file train/dog/
  find . -maxdepth 1 -type f -name 'dog*'| xargs -I file mv file test/dog/


# Now I remove them from the 'catdog' folder...

# finally use this to remove the " . " 's after every "dog" or "cat.
python
>>> import os
>>> for filename in os.listdir("."):
...   os.rename(filename, filename[0:3]+filename[4:])
