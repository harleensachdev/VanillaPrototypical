# FS1-EcoAcousticAlarmDetection

Vanilla Prototypical Network to classify ecological audio recordings into seven categories: alarm, non-alarm, background, insect call, rain, low frequency sounds and high frequency sounds. The model begins by converting MP3 or WAV files into Mel spectrograms and, for each episode, randomly splits samples into a support set (5 samples per class), query set (6 samples per class), and test set (30 samples per class). Using an episodic batch sampler, 100 training episodes are generated. A CNN encoder with four convolutional blocks extracts embeddings from spectrograms, optimized via the Adam optimizer and cross-entropy loss. These embeddings are used by a Prototypical Network, which computes class prototypes from the support set and compares them to query embeddings using Cosine distance, converting distances into log-probabilities for classification.During evaluation, the model processes the test set over 100 episodes, extracting embeddings and producing final predictions using only the Prototypical Network.

This model achieves an F1 of 94.2 with unseen data.
# VanillaPrototypical
# VanillaPrototypical
