# MultiModal_Dense_Video_Captioning

![Screenshot](CV_Methodology_Diagram (1).png)

## Usage

```bash
git clone --recursive https://github.com/porcupine12345/MultiModal_Dense_Video_Captioning.git
```

Download features I3D (17GB), VGGish (1GB) and put in `./data/` folder (speech segments are already there). You may use `curl -O <link>` to download the features.

Link for I3D features : https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/mdvc/sub_activitynet_v1-3.i3d_25fps_stack24step24_2stream.hdf5


Link for VGGish features : https://a3s.fi/swift/v1/AUTH_a235c0f452d648828f745589cde1219a/mdvc/sub_activitynet_v1-3.vggish.hdf5

```
# MD5 Hash
a661cfe3535c0d832ec35dd35a4fdc42  sub_activitynet_v1-3.i3d_25fps_stack24step24_2stream.hdf5
54398be59d45b27397a60f186ec25624  sub_activitynet_v1-3.vggish.hdf5
```

Setup `conda` environment. Requirements are in file `conda_env.yml`

```bash
# it will create new conda environment called MultiModal_Dense_Video_Captioning
'' on your machine
conda env create -f conda_env.yml
conda activate MultiModal_Dense_Video_Captioning

# install spacy language model. Make sure you activated the conda environment
python -m spacy download en
```

## Train and Predict

Run the training and prediction script. It will, first, train the captioning model and, then, evaluate the predictions of the best model in the learned proposal setting. It will take ~24 hours (50 epochs) to run on a 2080Ti GPU. Please, note that the performance is expected to reach its peak after ~30 epochs.

```bash
# make sure to activate environment: conda activate MultiModal_Dense_Video_Captioning

# the cuda:1 device will be used for the run
python main.py --device_ids 1
```



```
# MD5 Hash
55cda5bac1cf2b7a803da24fca60898b  best_model.pt
```
