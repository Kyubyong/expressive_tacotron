# A TensorFlow Implementation of Expressive Tacotron

This project aims at implementing the paper, [Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron](https://arxiv.org/abs/1803.09047), to verify its concept. Most of the baseline codes are based on my previous [Tacotron implementation](https://github.com/Kyubyong/tacotron).

## Requirements

  * NumPy >= 1.11.1
  * TensorFlow >= 1.3
  * librosa
  * tqdm
  * matplotlib
  * scipy

## Data

<img src="https://image.shutterstock.com/z/stock-vector-lj-letters-four-colors-in-abstract-background-logo-design-identity-in-circle-alphabet-letter-418687846.jpg" height="200" align="right">

Because the paper used their internal data, I train the model on the [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)

LJ Speech Dataset is recently widely used as a benchmark dataset in the TTS task because it is publicly available. It has 24 hours of reasonable quality samples.

## Training
  * STEP 0. Download [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/) or prepare your own data.
  * STEP 1. Adjust hyper parameters in `hyperparams.py`. (If you want to do preprocessing, set `prepro` True`.
  * STEP 2. Run `python train.py`. (If you set `prepro` True, run `python prepro.py` first)
  * STEP 3. Run `python eval.py` regularly during training.

## Sample Synthesis

I generate speech samples based on the same script as the one used for the original [web demo](https://google.github.io/tacotron/publications/end_to_end_prosody_transfer/). You can check it in `test_sents.txt`.

  * Run `python synthesize.py` and check the files in `samples`.


## Samples

16 sample sentences in the first chapter of the original [web demo](https://google.github.io/tacotron/publications/end_to_end_prosody_transfer/) are collected for sample synthesis. Two audio clips per sentence are used for prosody embedding--reference voice and base voice.
Mostly, those two are different in terms of gender or region. The samples below look like the following:

* 1a: the first reference audio
* 1b: sample embedded with 1a's prosody
* 1c: the second reference audio (base)
* 1d: sample embedded with 1c's prosody

Check out the samples at each steps.

* [130k steps](https://soundcloud.com/kyubyong-park/sets/expressive_tacotron_130k)
* [420k steps](https://soundcloud.com/kyubyong-park/sets/expressive_tacotron_420k)

## Analysis
  * Hearing the results of 130k steps, it's not clear if the model has learned the prosody.
  * It's clear that different reference audios cause different samples.
  * Some samples are worthy of note. For example, listen to the four audios of no.15. The stress of "right" part was obvious transferred.
  * Check out no.9, reference audios of which are sung. They are fun.

## Notes

  * Because this repo focuses on investigating the concept of the paper, I did not follow some details of the paper.
  * The paper used phoneme inputs, whereas I stuck to graphemes.
  * Instead of the Bahdanau attention, the paper used the GMM attention.
  * The original audio samples were obtained from wavenet vocoder.
  * I'm still confused what the paper claims to be a prosody embedding can be isolated from the speaker.
  * For prosody embedding, the authors employed conv2d. Why not conv1d?
  * When the reference audio's text or sentence structure is totally different from the inference script, what happens?
  * If I have time, I'd like to implement their 2nd paper: [Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis](https://arxiv.org/abs/1803.09017)

  April 2018,
  Kyubyong Park
