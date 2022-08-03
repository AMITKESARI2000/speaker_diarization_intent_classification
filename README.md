#### Name: Amit Kesari (ML Team)
##### Mentor : Mr. Amandeep Arora

## Problem Statement
- To perform Speaker Diarization in audio calls and get the speaker turns and embeddings of various speakers
- Validation and tuning of pre-trained/fine-tuned pipeline on generic dataset
- Single Label Text Classification of Vape.csv
- Relevant/Non_relevant Intent Classification on Retail.csv

## Challenges
- Overlaps, interruptions, background noise and overtalk, so cutting the audio between sentences isn't trivial
- Speakers have to be discovered dynamically. Subtle difference between two voice speakers may lead to a single cluster
- Research and experiments with different pipelines of speaker diarization and setting them up. 
- Prepping up the audio dataset files and folders along with getting the ground-truths
- Fine-Tuning models and pipeline (pyannote) for SD on datasets is difficult and tedious and no standard way like in Text Classification.
- For Text classification, getting out differentiating factors and mislabeled classes.

## 1. Work Done (SD):
Diarization Error Rate (DER):
- pyannote.audio
- NeMo (oracle VAD)
- NeMo (SD+ASR)
- SpeechBrain
- diart (Online)


Dataset (along with .rttm files)
- AMI Meeting Corpus(English): a multi-modal data set consisting of 100 hours of meeting recordings. The meetings were recorded in English using three different rooms with different acoustic properties. AMI provides an official {training, development, test} partition of the Mix-Headset audio files into 136, 18 and 16 each. It has ~20% overlaps and ~4 speaker per video

- VoxConverse 0.0.2(Multilingual): an audio-visual diarisation dataset consisting of over 50 hours of multi-speaker clips of mostly english human speech, extracted from YouTube videos. VoxConverse provides a proper development and test set of 216 and 232 audio files each. Test set has ~3.1% overlaps and ~6 speakers per video


## 2. Work Done (Classification 1: Single Label Text Classification):
Sklearn classification models:
- ComplementNB
- MultinomialNB
- LogisticRegression
- LinearSVC
- Random Forest

Transformer Based Models:
- bert-base-uncased
- xlnet-base-cased
- roberta-base
- distilbert-base-uncased
- mpnet-base
- ms-marco-MiniLM-L-12-v2
- deberta-base
- deberta-v2-xlarge
- electra-base-emotion
- ms-marco-electra-base

Dataset :
train_dropdups.csv: 4840 entries
val_dropdups.csv: 537 entries
test_dropdups.csv: 832 entries

## 3. Work Done (Classification 2: Intent Classification):
- bert-base-uncased
- xlnet-base-cased
- roberta-base
- distilbert-base-uncased (torch 1.7.1, transformer 3.3.1)
- mpnet-base
- ms-marco-MiniLM-L-12-v2
- deberta-base
- distilroberta-base
- distilroberta-base (torch 1.7.1, transformer 3.3.1)
- Update after end eval: (torch 1.7.1, transformer 3.3.1)
- roberta-base
- distilroberta-base
- distilbert-base-uncased

Dataset :
train_final_retail.csv: 140902 entries
val_final_retail.csv: 28180 entries
test_final_retail.csv: 18788 entries

## Research Papers & References Read
- Bredin, Hervé, et al. "Pyannote. audio: neural building blocks for speaker diarization." https://arxiv.org/pdf/1911.01255.pdf  
- Coria, Juan M. and Bredin et al. Overlap-Aware Low-Latency Online Speaker Diarization Based on End-to-End Local Segmentation, 2021 https://arxiv.org/pdf/2109.06483.pdf 
- McCowan, Iain, et al. "The AMI meeting corpus." 2005 https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.95.6326&rep=rep1&type=pdf 
- Chung, Joon Son, et al. "Spot the conversation: speaker diarisation in the wild."(2020) https://arxiv.org/pdf/2007.01216.pdf 
- Yu, Fan, et al. "M2MeT: The ICASSP 2022 multi-channel multi-party meeting transcription challenge." https://arxiv.org/pdf/2202.03647.pdf 
- Koluguri, Nithin Rao, Taejin Park, and Boris Ginsburg. "TitaNet: Neural Model for speaker representation with 1D Depth-wise separable convolutions and global context." https://arxiv.org/pdf/2110.04410.pdf 
- Nauman Dawalatabad, Mirco Ravanelli, et al. “Ecapa-tdnn embeddings for speaker diarization” 2021 https://arxiv.org/pdf/2104.01466.pdf 
- Park, T.J., Kanda, N., Dimitriadis, D., Han, K.J., Watanabe, S. and Narayanan, S., 2022. A review of speaker diarization: Recent advances with deep learning. https://arxiv.org/pdf/2101.09624.pdf 
- H.Zhang et al. Effectiveness of Pre-training for Few-shot Intent Classification,2021 https://arxiv.org/pdf/2109.05782.pdf 
- NLP Specialization Course on Coursera: https://www.coursera.org/specializations/natural-language-processing 
- Deep Learning Specialization on Coursera: https://www.coursera.org/specializations/deep-learning 


