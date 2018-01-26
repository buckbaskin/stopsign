# Video Processing

How to go from a video (or videos) to a trained classifier.

## Split the video into its component image frames

Use `ffmpeg`.

## Generate vectors from images

Use `manually_annotate_images2.py`

This program previews randomly selected images from the dataset. The user selects either stopsign or not stopsign, and (if stopsign) selects the region containing the stopsign. All contained features are saved as stopsign vectors and the rest are saved as not stopsigns.

Example Output file: `all_500.csv`

## Clean up data

Use `clean_up_man_labels2.py`

This program adjust some of the silly decisions I made when I wrote the first program in terms of how it saves CSV files.

Example Output file: `clean_500.csv`

## Split into positive and negative data files

Use `split_pos_neg.py`

This ammounts to subsampling the data as well as moving to more manageable file sizes. The negative data is 5 random samples of negative data equivalent in size to the positive dataset.

Example Output files: `positive_500.csv` and `negative_500_0.csv`

## Make byte-wise data bit-wise data

Use `bytes_to_bits.py`

This is where manageable file sizes come in handy.

Example Output files: `positive_bits_500.csv` and `negative_bits_500_0.csv`

## Optimize the Learning Algorithm

Example: use `bdt_optimize.py`

The script specifies model parameters to iterate over. The algorithm is then trained a couple different times and the performance is compared.