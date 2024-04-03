# Mel-Spectogram

### 스펙토그램이란?

- 보통 딥러닝 모델은 raw 오디오 파일을 인풋으로 받지 않고 스펙토그램으로 변환하여 학습
- 스펙토그램은 오디오 wave의 스냅샷이며 이미지 파일이며 CNN 계열 구조의 input으로 넣어 학습
- 시간에 따라 변화하는 소리 또는 다른 신호의 주파수 스펙트럼을 시각적으로 나타낸 것입니다. 일반적인 스펙트로그램과 다른 점은 주파수가 사람 귀의 다른 주파수에 대한 반응을 근사하는 mel-spectogram 척도로 변환
- 오디오 신호를 단시간 프레임으로 변환하고, 이 프레임에 Fourier에 변환을 적용하여 주파수 스펙트럼을 얻은 다음, 얻어진 파워를 mel-spectogram 척도에 매핑하는 것을 포함
- 축에는 시간, 다른 축에는 멜-주파수가 있고, 색깔은 특정 주파수의 존재를 나타내는 2D 표현

### 스펙토그램  생성 방법?

- Fourier Transform을 활용하여 sound signal로 부터 스펙토그램을 생성하다

- **Fourier Transform**

[But what is the Fourier Transform?  A visual introduction.](https://youtu.be/spUNpyF58BY)

- 요약하면 입력 신호를 다양한 주파수를 가지는 주기 함수들로 분해
- 하지만 음성 신호에 그냥 fft를 사용해 버리면 **각각의 주파수 성분이 언제 존재하는지는 알 수 없음**
- 음성 신호에 Fourier 변환을 하는 것은 시간정보가 손실되어서 음성합성 & 인식에 이용하기에는 적합x

```python
import librosa
from scipy.io import wavfile
import librosa.display
import matplotlib.pyplot as plt

# Load the audio file
AUDIO_FILE = './audio.wav'
samples, sample_rate = librosa.load(AUDIO_FILE, sr=None)

sample_rate, samples = wavfile.read(AUDIO_FILE)

# x-axis has been converted to time using our sample rate. 
# matplotlib plt.plot(y), would output the same figure, but with sample 
# number on the x-axis instead of seconds
plt.figure(figsize=(14, 5))
librosa.display.waveplot(samples, sr=sample_rate)
```

![mel-sepctogram_1](/docs/Img/mel-sepctogram_1.png)

![https://blog.kakaocdn.net/dn/RxcG2/btqNQagZ7j0/bKms8bcfYDUvA0TvTKXbkK/img.png](https://blog.kakaocdn.net/dn/RxcG2/btqNQagZ7j0/bKms8bcfYDUvA0TvTKXbkK/img.png)

### Audio Signal Data

- 오디오 데이터는 특정 시간 주기 간격으로 사운드 웨이브를 샘플링하여 각 샘플에서 웨이브의 강도나 진폭을 측정함으로써 얻어짐. 오디오의 메타데이터는 샘플링 레이트를 알려주는데, 이는 초당 샘플의 수입니다.
- 오디오가 파일로 저장될 때는 압축 형식으로 저장
- 파일이 로드될 때, 압축 해제되고 Numpy 배열로 변환됩니다. 시작한 파일 형식과 관계없이 이 배열은 동일
- 메모리에서 오디오는 각 타임스텝에서의 진폭을 나타내는 숫자의 시계열로 표현됩니다. 예를 들어, 샘플 레이트가 16800이라면, 1초 길이의 오디오 클립은 16800개의 숫자를 가집니다. 측정은 고정된 시간 간격에서 이루어지기 때문에, 데이터는 진폭 숫자만을 포함하고 시간 값은 포함하지 않습니다. 샘플 레이트를 알고 있다면, 각 진폭 숫자 측정이 어느 시간 순간에 이루어졌는지 알 수 있

![mel-sepctogram_2](/docs/Img/mel-sepctogram_2.png)

- 비트 깊이는 각 샘플의 진폭 측정값이 가질 수 있는 가능한 값의 수를 알려줍니다. 예를 들어, 비트 깊이가 16이라면 진폭 수치는 0부터 65535(2¹⁶ — 1) 사이가 될 수 있습니다. 비트 깊이는 오디오 측정의 해상도에 영향을 미칩니다 — 비트 깊이가 높을수록 오디오 충실도가 더 좋아집니다.

### **STFT(Short Time Fourier Transform)**

- 음성 데이터를 시간 단위로 짧게 쪼개서 FFT 를 하는 행위
- librosa 라이브러리에서 지원

[librosa.stft — librosa 0.10.1 documentation](https://librosa.org/doc/latest/generated/librosa.stft.html)

- **n_fft** : length of the windowed signal after padding with zeros.

2. **hop_length :** window 간의 거리

3. **win_length :** window 길이

![mel-sepctogram_3](/docs/Img/mel-sepctogram_3.png)

- 음성 신호를 인식할때 주파수를 linear scale로 인식X
- 낮은 주파수를 높은 주파수보다 더 예민하게 받아들인다고 한다. 즉 500 ~ 1000 Hz 가 바뀌는건 예민하게 인식하는데 10000Hz ~ 20000 Hz가 바뀌는 것은 잘 인식 못함
- 10000hz ~ 20000hz는 mel scale로 변환하여 인식
    - Mel(f) = 2595 log(1+f/700)
    
    ![mel-sepctogram_4](/docs/Img/mel-sepctogram_4.png)
    

![mel-sepctogram_5](/docs/Img/mel-sepctogram_5.png)