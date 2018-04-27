# Speech Emotion Recognition using DNN-ELM

Emotion Signals are divided into segments and then extract the segment-level features to train a DNN. The trained DNN computes the emotion state probability distribution for each segment. Statistics of the segment-level emotion state probabilities to determine emotions are applied at the utterance-level. Extreme Learning Machine (ELM) is employed to conduct utterance-level emotion classification. It used only improvised data in Interactive Emotional Dyadic Motion Capture (IEMOCAP) database.

For low-level acoustic features, In [1], Authors extract 12-dimensional Mel-frequency cepstral coefficients (MFCC) with log energy, pitch period, harmonics to noise ratio (HNR), and derivatives. And F0 (pitch), voice probability, zero-crossing rate, and their first-time derivatives are applied in [2]. 

## Datasets
* Interactive Emotional Dyadic Motion Capture (IEMOCAP) database is required to run this code.
* It used only improvised data for context-independent situation evaluation.
* Cross validation scheme is applied for speaker-independent manner.

## Dependencies
* Librosa for some low-level acoustic features extraction
* Tensorflow for Deep Neural Networks
* scikit-learn for acurracy evaluation

## References
* [1] K. Han, D. Yu, and I. Tashev, "Speech emotion recognition using deep neural network and extreme learning machine," in Proc. Interspeech, 2014.
* [2] J. Lee and I. Tashev, "High-level Feature Representation using Recurrent Neural Network for Speech Emotion Recognition," in Proc. Interspeech, 2015.


