# SeeAction

### Please access [Table_results](https://github.com/zzzzdh/SeeAction/tree/main/Table_results) to check all experiment results in this paper.

### Check examples output by our model [here](https://github.com/zzzzdh/SeeAction/blob/main/examples/README.md)

## Dataset

### Please download the dataset [here](https://drive.google.com/file/d/1KXA4SDEfrFP-1GGYtmqE63eWSBTgxDKE/view?usp=sharing).
You can also construct your own dataset by the following instructions.

Note: Please revise all directory in the code to your own directory.

1. Prepare video data
2. Run `extract_frame.py` to split video to frames. This step uses ffmpeg. You can install it by `apt-get install ffmpeg`.
3. Run `com_diff.py` to compute change regions and crop the change regions.
4. Annotate labels by our tool `label_tool.py`

## Train
1. Modify `config.py`
2. Start training `train.py`

### We provide HD figures for this paper.

| ![](/Fig/structured_example.jpg) | 
|:--:| 
| *Fig. 1: Examples of structured HCI actions*|

| ![](/Fig/preprocessing.jpg) | 
|:--:| 
| *Fig. 2: Compute similarity map and detect change region*|

| ![](/Fig/model.jpg) | 
|:--:| 
| *Fig. 3: Model architecture*|

| ![](/Fig/failure_location.jpg) | 
|:--:| 
| *Fig. 4:  Failure examples of location prediction*|

| ![](/Fig/failure_lstm.jpg) | 
|:--:| 
| *Fig. 5: Failure example of traditional video captioning*|

| ![](/Fig/failure_zoom.jpg) | 
|:--:| 
| *Fig. 6: Failure example of Zoom meeting*|

| ![](/Fig/pipeline.jpg) | 
|:--:| 
| *Fig. 7: Our Screencast-to-ActionScript Tool*|

