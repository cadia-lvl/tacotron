## Setup
* Install requirements: `pip install -r requirements.txt`
* Before training, the data has to be preprocessed via `preprocess.py`
* Now you can start training

## Other notes and Issues
* If you run into filewatcher issues, blacklist datasets, training, virtualenv and `.git`.

## Support
The following datasets have support in this repository:
* LJSpeech-1.1
* TTS_icelandic_Google_m
* unsilenced_icelandic
Currently, to make sure no data is lost, additions have to be made to `preprocess.py` to be able to use other datasets. This has already been done for a version of TTS_icelandic_Google_m where `unsilencer.py` was applied to the dataset.


## Example of running from scratch
1. Preprocess the data
    * We assume there is a supported dataset at the absolute path `path_to_dataset_base = <input_dir>`. and `<output_dir>` is relative to `/home/<user>`, meaning that selecting `<output_dir>` as `work/processed` results in the absolute path `/home/<user>/work/processed`. 
    * To preprocess , run `python3 preprocess.py --input_dir=<input_dir> --output_dir=<output_dir> --dataset_name=<dataset_name>` where `<dataset_name>` is one of the supported datasets.
    * This results in the processed data being stored at `/home/<user>/<output_dir>`.
2. Training
    * Can only be done on a pre-processed dataset. We assume that the processed data is located at `/home/<user>/<input_dir>`. Training data will be stored at `/home/<user>/<output_dir>`. The dataset name is selected again here but the `model_name` variable has to be set as well. We do this to be able to seperate 2 different models inferring on the same dataset but with different hyper parameters. There are some other arguments that are optional or required that can be seen in the source of `runner.py`
    * To start training, run `python3 runner.py --input_dir=<input_dir> --output_dir=<output_dir>  --model_name=<model_name>...`
    * This eventually results in the training data being stored at `home/<user>/<output_dir>/<model_name>`. Under that directory you should find `logs`, `model` and `samples`
3. Synthesize
    * We assume that model-data is stored at `/home/<user>/<input_dir>` for `<model_name>`. A `<restore_step>` and `<text>` has to be supplied.
    * To synthesize run `python3 synthesize.py --input_dir=<input_dir> --restore_step=<restore_step> --text=<text>`. This results in the synthesized data being stored at `/home/<user>/<input_dir>/<model_name>/synthesized`

