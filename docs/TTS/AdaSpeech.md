# TTS - AdaSpeech

논문: 

[https://arxiv.org/pdf/2103.00993v1.pdf](https://arxiv.org/pdf/2103.00993v1.pdf)

## 기본 정보

- **논문 제목**: ADASPEECH: ADAPTIVE TEXT TO SPEECH FOR CUSTOM VOICE
- **모델 이름** : AdaSpeech V1

## 주제를 선정한 이유

이전부터 관심 있었던 주제 중 하나이다. 반복적인 업무를 대체하고 싶었던 마음이 제일 컸으며 광고, 공지 등 간단한 발표 작업을 AI로 대체하고 싶었다. 특히, 요즘 뜨고 있는 주제 중 하나가 AI가수처럼 자연스러운 발성, 발음 등 구현하는 방법에 대해 관심이 더 커져 해당 주제를 선정하게 되었다.


## 개념

TTS는 Text to Speech의 줄임말로 최근에 인터넷 방송 도네이션 목소리가 이전 AI 목소리가 아닌 사용자의 목소리 특징 등을 학습하여 사용자의 고유의 목소리로 TTS를 하는 기술이다.  그러나 TTS 기술을 이용하기 위해서는 직접 문장을 읽은 12시간 가량의 음성이 필요해, 다양한 목소리를 만드는 데 어려움이 있다.



## 도입

TTS - Text to Speech 작업은 너무나도 다양한 요소에 영향을 미치기에 정확한 목소리 복제란 복잡하다. 목소리, 사투리, 녹화 당시 배경음 등에 따라 목소리 클론의 정확도가 달라지기 마련이다.

기존에 해왔던 이미지 데이터 처리와는 다르게 고려해야 되는 부분이 

## 방법론

### 학습 방법

기본적인 학습 방법은 아래와 같다.


1. Pre-Training
    - ensure the TTS model covers diverse text and speaking voices that are helpful for adaptation
2. fine-tuning
    - adapted on a new voice by fine-tuning (a part of) the model parameters on the limited adaptation data with diverse acoustic conditions
3. Inference

- ***Pre-training 단계에서는 대량의 데이터셋을 이용하여 학습시키고 파인튜닝 작업으로 새로운 목소리를 자연스럽게 도와주는 역할***

### AdaSpeech

AdaSpeech는 현재 버전4 까지 나온 것으로 보이며 공개되어 있는 AdaSpeech 모델은 보기가 어려워 AdaSpeech를 별도로 구성했던 사람들의 코드를 참고하며 진행하고 있다.

도입에서 말했던 내용 중에 사용자의 녹음 환경, 발음, 사투리 등을 반영하기 위해 아래와 같은 방식으로 모델링을 한다. 


- Acoustic condition Modeling
    - 두개의 Acoustic 인코더를 사용하여 utterence-level (목소리 강약, 톤 등) 벡터와 phoneme(비슷한 발음)레벨 벡터를  추출하여 mel-spectogram 디코더에 인풋으로 입력된다.
    - 이 방법을 통해 녹음 환경에 제약을 받지 않게 하고 정규화하는데 도움을 준다.
    - 아웃풋으로는 인코더에 입력했던 목소리 강약 등의 utterence-level vector를 speech로부터 추출하고 phoneme encoder로 phoneme-level vector를 추출한다.

- Conditional layer normalization
    - 소량의 파라미터로 품질은 보증하면서 파인튜닝을 하기 위해서는 melspectogram decoder 레이어 정규화를 수정
        - speaker embedding을 조건 정보를 활용하여 레이어 정규화에 scale 과 bias vector 생성
    - In fine-tuning, we only adapt the parameters related to the conditional layer normalization. In this way, we can greatly reduce adaptation parameters and thus memory storage compared with fine-tuning the whole model, but maintain high-quality adaptation voice thanks to the flexibility of conditional layer normalization.

### 2. AdaSpeech 모델 구조

- 다양한 목소리 등 학습시키기 위함
- conditional layer normalization in decoder를 활용하여 다양한 사람의 목소리를 적당한 메모리를 사용할 수 있도록 지원

![adaSpeech](/docs/Img/adaSpeech.png)

In custom voice, the adaptation data can be spoken with diverse prosodies, styles, and accents, and can be recorded under various environments, which can make the acoustic conditions far different from those in source speech data
추가로, 사용자의 녹음 환경, 발음 등 너무 다양하여 genaralization(정규화) 작업이 어렵다.

To better model the acoustic conditions with different granularities, we categorize the acoustic
conditions in different levels as shown

- speaker level, the coarse-grained acoustic conditions to capture the overall characteristics of a speaker;
- utterance level, the fine-grained acoustic conditions in each utterance of a speaker;
- phoneme level, the more fine-grained acoustic conditions in each phoneme of an utterance, such as accents on specific phonemes, pitches, prosodies, and temporal environment noises

*Since speaker ID (embedding) is widely used to capture speaker-level acoustic conditions in multi-speaker scenarios (Chen et al., 2020), speaker embedding is used by default*