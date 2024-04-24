from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# 이미지 경로 설정
img_path = "C:\\Users\\김민정\\Desktop\\초음파_modified CWT\\img\\RAW\\RAW_h\\ap_240402_tek0100ALL_RAW.png"

# 이미지를 로드하고 크기 조정
img = image.load_img(img_path, target_size=(1500, 800))

# 이미지를 배열로 변환
img_array = image.img_to_array(img)

# 이미지 배열의 차원 확장 (모델이 기대하는 배치 크기를 만듦)
img_array = np.expand_dims(img_array, axis=0)

# 이미지 확인
plt.imshow(np.squeeze(img_array).astype('uint8'))
plt.title("Resized Image")
plt.show()

# 모델 입력 전 이미지 배열 정규화 (이미 ImageDataGenerator에서 rescale을 1./255로 설정했다면 생략 가능)
img_array /= 255.0

# 이제 img_array를 모델에 입력으로 사용할 수 있습니다.
# 예: predictions = model.predict(img_array)