# drum-aware4beat
This repo contains the source codes for paper titled
“Drum-Aware Ensemble Architecture for Improved Joint Musical Beat and Downbeat Tracking”.
| [**Paper (arXiv)**](https://arxiv.org/abs/2106.08685) | [**Github**](https://github.com/SunnyCYC/drum-aware4beat) | [**Pretrained Models**](https://github.com/SunnyCYC/drum-aware4beat/tree/main/pretrained_models) |

This paper presents a novel system architecture that integrates blind source separation with joint beat and downbeat tracking in musical audio signals. The source separation module segregates the percussive and non-percussive components of the input signal, over which beat and downbeat tracking are performed separately and then the results are aggregated with a learnable fusion mechanism. This way, the system can adaptively determine how much the tracking result for an input signal should depend on the input’s percussive or non-percussive components. Evaluation on four testing sets that feature different levels of presence of drum sounds shows that the new architecture consistently outperforms the widely-adopted baseline architecture that does not employ source separation.

## usage of this repo
### inference:
* Pretrained models mentioned in the paper are provided in [pretrained_models](https://github.com/SunnyCYC/drum-aware4beat/tree/main/pretrained_models). You may run Inference.py to see how it works on the assigned test song. 
* To test on you own audio, replace the audio_file_path in inference.py with your test song path should work. 

### training with your own data:
* dataset preparation:
    * **Note that this repo only include 10 songs from GTZAN [1] to demonstrate the usage. You may directly run the training scripts (e.g. [train_DA1.py](https://github.com/SunnyCYC/drum-aware4beat/blob/main/train_DA1.py)) to see how it work.** To train your model with more datasets, please see [this repo](https://github.com/SunnyCYC/aug4beat) for how to organize the dataset folders and file names. 
    * After uploading your datasets to datasets/original/ following the instructions, you may run the follow scripts to get training data prepared:
        * [traintest_split.py](https://github.com/SunnyCYC/aug4beat/blob/main/traintest_split.py)
        * [source_separation_aug4beat.py](https://github.com/SunnyCYC/aug4beat/blob/main/source_seperation_aug4beat.py)
        * [prepare_dataset.py](https://github.com/SunnyCYC/aug4beat/blob/main/prepare_dataset.py)
* training:
    * Once your dataset is prepared, the dataset information will be saved in train_dataset.ab and valid_dataset.ab in datasets folder.
    * Run script [train_bsl.py](https://github.com/SunnyCYC/drum-aware4beat/blob/main/train_bsl.py), [train_DA1.py](https://github.com/SunnyCYC/drum-aware4beat/blob/main/train_DA1.py), or [train_DA2.py](https://github.com/SunnyCYC/drum-aware4beat/blob/main/train_DA2.py) could start the training. You may also modify the experiment parameters within main() of these scripts.
* hmm optimization:
    * As each trained model may have its prefered parameters for HMM post processing, you may follow HMM optimzation section of [this repo](https://github.com/SunnyCYC/aug4beat) to find best HMM for your trained model. 

## Reference
[1] G. Tzanetakis and P. Cook, “Musical genre classification of audio signals,” IEEE Trans. Speech and Audio Processing, vol. 10, no. 5, pp. 293–302, 2002.
