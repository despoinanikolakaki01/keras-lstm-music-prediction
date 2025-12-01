# Keras LSTM Music Generator
Learn to generate music with LSTM neural networks in Keras

# Requirements
Python 3.x

Keras and music21 

GPU strongly recommended for training

# Info
After reading Sigurður Skúli's towards data science article ['How to Generate Music using a LSTM Neural Network in Keras'](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5) - I was astounded at how well LSTM classification networks were at predicting notes and chords in a sequence, and ultimately then how they could generate really nice music. 

Sigurður's approach had some really nice and useful functions for parsing the data, creating dictionaries to translate between notes and class labels, and then using the trained model to generate pieces of music. 


1. The Notes and Chords in the sequence (just referred to as 'notes' from here on)
2. The offsets of the note from the previous one (offset of the note from the start of the midi minus the current base (previous value))
3. The durations of the notes in the sequence

The above also serve as three separate outputs to the network.

Thus, three tasks are trained. Classifying the next note, its offset, and its duration.

Prior-conditioned generation (this work vs. original):
This repo adds a prior-conditioning mode to the original Keras-LSTM generator. The script predict_with_prior.py extracts a short real MIDI fragment (notes, offsets, durations) and use it as a seed "prior" that is prepended to the generated output. In practice, the predictor parses a chosen MIDI phrase and asks the trained LSTM to continue it — producing a continuation that is stylistically tied to the prior fragment. By contrast the original predict-tf2.py chooses a random seed from the training patterns and generates music from that seed without explicitly including a real prior. Notes:
The prior script intentionally generates a longer continuation (700 notes in the version above) to show extended continuation behavior.
Be sure the predictor's sequence_length matches the sequence length used at training time (mismatched sequence lengths will cause shape errors or incorrect behaviour). The training script lstm-new-tf2.py uses sequence_length = 100, so use predict_with_prior_last.py (which uses 100) when loading that model; predict_with_prior.py uses sequence_length = 20 and must match a model trained with that smaller window.
Suggested small README additions (optional)

Add a short example showing how to run prior-conditioned generation:
Add a note about weights/data:
“Prior-conditioned generation requires the same model weights used in training. If pretrained weights are not present, run python lstm-new-tf2.py to train (200 epochs, batch_size=64, Adam lr=0.001).”
Would you like me to:

A) Insert the above README paragraph into README.md and commit it, or
B) Just give the text so you paste it yourself?
If A, tell me whether to add it near the top or under the “Training” / “To Do” section.

GPT-5 mini • 0x


# Model Diagram
This is the best model I have found so far:
![LSTM Model Diagram](model_plot.png)
Note: the number of outputs for the three final layers differ depending on what was detected during parsing of the files. For example, if you parse 100 midi files consisting of only three chords, the notes_output layer will only have three outputs. Thus, depending on what kind of music you're training on, the number of softmax neurons will change dynamically. 


# Training 
Everything in this repo can be run as-is to train on classical piano pieces:

1. Run train.py for a while until note/chord loss is below 1
2. Set line 155 in generate-music.py to the name of the .HDF5 file that was last saved (the model weights)
3. Run generate-music.py to generate a midi file from the model

## Using your own data
Since most midis are multi-track (eg. 2x piano, left and right hand), and this model only supports one - run convertmidis.py to merge all tracks into one

Change line 53 in train.py to specify where your .midi files are stored

# TensorFlow 2.0
To run with TensorFlow 2.0 use lstm-new-tf2.py

# To Do
I need to see what happens when the offset and duration problems are regression with MSE. Currently, we do classification of a library of classes (ie. unique offsets and durations) detected from the dataset during Midi parsing

Enable multi-track training and prediction, currently we just merge all midi tracks into one
Upload some more examples of music
Try and experiment with smaller models that are quicker to train but still produce good results
Try out some different loss functions, Adam seems best so far
Maybe experiment with adding a genre classification network branch so the model doesn't need curated data as input
Clean up the code, it's a bit messy in places

# Changing instruments
I hope to update the model to learn to predict the instrument, but at the moment I just use https://onlinesequencer.net/ if I want to hear it played by something other than a piano

You can train the network on any instrument, all it cares about are the notes and their offset and duration. But, that said, the network will set each note's instrument as piano. This can be changed via lines 265 and 282 which set notes and chords to piano respectively in generate-music.py

## Publishing this project

- I plan to publish this repository without large datasets or pretrained weights. The `classical-piano` and `classical-piano-type0` folders should be downloaded separately or provided as a small sample in `samples/`.
- Pretrained weights are not included. To reproduce the experiments, run `lstm-new-tf2.py` (200 epochs, batch size 64, Adam lr=0.001) to create weight files in `weights_maestro/`.

### Quickstart (run minimal example)

1. Create and activate a python environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. To retrain (example):

```powershell
python lstm-new-tf2.py
```

3. To generate music (after you have weights):

```powershell
python predict_with_prior_last.py
```

### Data

Do NOT commit the full MAESTRO dataset into this repository. Download the MAESTRO dataset from the official source and place it under `data/maestro/` or `midis/` according to the scripts' expectations. See `convertmidis.py` for how to preprocess multi-track MIDI into a single-track format.

If you have small example MIDI or audio samples, put them in `samples/` (these can be committed). Large model weights and audio should be uploaded as Releases or stored using Git LFS / external archive (Zenodo / Hugging Face / Google Drive) and linked from here.

