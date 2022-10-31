import keras
import sklearn
import tensorflow as tf
from keras import activations, layers, losses, callbacks, optimizers, metrics, initializers

import data

# 서버에서 사용하는 2개의 GPU 중, 2번째 GPU만을 사용하여 연산을 수행하도록 합니다.
gpu_list = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(gpu_list[1], 'GPU')


class Model:
    """
    학습 모델 클래스입니다.
    """

    def __init__(self, model_name: str, data_path: str):
        """
        :param model_name: 모델의 이름(파일 저장용)
        :param data_path: csv 데이터셋 경로
        """
        self.model_name = model_name
        self.model_path = f"{model_name}.tf"
        self.input_shape = (29,)
        length = self.input_shape[0]
        self.model = keras.Sequential([
            layers.Input(self.input_shape),
            layers.BatchNormalization(),
            layers.Dense(units=29, activation=activations.selu,
                         kernel_initializer=initializers.initializers_v2.LecunNormal(seed=42)),
            layers.AlphaDropout(rate=0.3, seed=42),
            layers.Dense(units=10, activation=activations.selu,
                         kernel_initializer=initializers.initializers_v2.LecunNormal(seed=42)),
            layers.AlphaDropout(rate=0.3, seed=42),
            layers.Dense(units=10, activation=activations.selu,
                         kernel_initializer=initializers.initializers_v2.LecunNormal(seed=42)),
            layers.AlphaDropout(rate=0.3, seed=42),
            layers.Dense(units=10, activation=activations.selu,
                         kernel_initializer=initializers.initializers_v2.LecunNormal(seed=42)),
            layers.AlphaDropout(rate=0.3, seed=42),
            layers.Dense(units=40, activation=activations.selu,
                         kernel_initializer=initializers.initializers_v2.LecunNormal(seed=42)),
            layers.AlphaDropout(rate=0.3, seed=42),
            layers.Dense(1, activation=activations.sigmoid)
        ])
        self.model.compile(optimizer=optimizers.Nadam(learning_rate=0.0001), loss=losses.binary_crossentropy,
                           metrics=[metrics.accuracy])
        x_data, y_data = data.load_and_preprocess_data(data_path)
        self.x_train, self.x_test, self.y_train, self.y_test = sklearn.model_selection.train_test_split(x_data, y_data,
                                                                                                        train_size=0.8,
                                                                                                        random_state=42)

    def train(self):
        """
        모델 훈련을 진행합니다.
        """
        self.model.fit(x=self.x_train, y=self.y_train, batch_size=2500, validation_data=[self.x_test, self.y_test],
                       epochs=1000000,
                       callbacks=[
                           callbacks.BackupAndRestore('./backup/', delete_checkpoint=True),
                           callbacks.CSVLogger(f'./logs/{self.model_name}.log', append=True),
                           callbacks.EarlyStopping(monitor='val_loss', patience=5000, restore_best_weights=True),
                           callbacks.TensorBoard(log_dir='tensorboard')
                       ])
        self.model.save(self.model_path)

    def evaluate(self):
        """
        모델 검증 결과를 출력합니다.
        """
        print(self.model.evaluate(self.x_test, self.y_test))

    def load_model(self, path: str, is_weight: bool):
        """
        지정된 경로로부터 모델 혹은 모델의 가중치 정보를 불러옵니다.
        """
        if is_weight:
            self.model.load_weights(path)
        else:
            self.model = keras.models.load_model(path)
