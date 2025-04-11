import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

seed = 22
tf.random.set_seed(seed)
np.random.seed(seed)
dataset = np.load("librispeech-train-100-clean-mfe-1sec.npz")
samples = dataset['features'][:, :40, :]
classes = dataset['speaker_labels'].astype(int)
threshold = 1448
counts = np.unique(classes, return_counts=True)[1]
keep_classes = np.where(counts >= threshold)[0]

filtered_samples, filtered_classes = [], []
for i in range(len(classes)):
    if classes[i] in keep_classes:
        filtered_samples.append(samples[i])
        filtered_classes.append(classes[i])

filtered_samples = np.array(filtered_samples)
filtered_classes = np.array(filtered_classes)
unique_classes = np.unique(filtered_classes)
class_mapping = {cls: idx for idx, cls in enumerate(unique_classes)}
filtered_classes = np.array([class_mapping[cls] for cls in filtered_classes])
X_train, X_temp, y_train, y_temp = train_test_split(
    filtered_samples, filtered_classes, test_size=0.3, random_state=seed)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=seed)

X_train = np.expand_dims(X_train, axis=-1)
X_val = np.expand_dims(X_val, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

teacher_dvec_model = tf.keras.models.load_model("d-vector-extractor-256.h5")
teacher_dvec_model.summary()

def build_student_model(input_shape, d_vector_dim=256):
    inputs = tf.keras.Input(shape=input_shape, name="input")
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(192, activation="relu", name="dense_1")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(192, activation="relu", name="dense_2")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(192, activation="relu", name="dense_3")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    d_vector = tf.keras.layers.Dense(d_vector_dim, activation="relu", name="d_vector")(x)

    return tf.keras.Model(inputs, d_vector, name="student_dnn_dvector")

class DVectorDistiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.sim_tracker = tf.keras.metrics.Mean(name="cosine_sim")

    @property
    def metrics(self):
        return [self.loss_tracker, self.sim_tracker]

    def train_step(self, data):
        x, _ = data
        with tf.GradientTape() as tape:
            teacher_dvec = self.teacher(x, training=False)
            student_dvec = self.student(x, training=True)
            teacher_dvec = tf.math.l2_normalize(teacher_dvec, axis=-1)
            student_dvec = tf.math.l2_normalize(student_dvec, axis=-1)
            cosine_sim = tf.reduce_sum(teacher_dvec * student_dvec, axis=-1)
            loss = 1 - tf.reduce_mean(cosine_sim)

        # Compute gradients and update weights
        gradients = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        # Update metrics
        self.loss_tracker.update_state(loss)
        self.sim_tracker.update_state(cosine_sim)
        return {"loss": self.loss_tracker.result(),
                "cosine_sim": self.sim_tracker.result()}

    def test_step(self, data):
        x, _ = data
        teacher_dvec = self.teacher(x, training=False)
        student_dvec = self.student(x, training=False)
        teacher_dvec = tf.math.l2_normalize(teacher_dvec, axis=-1)
        student_dvec = tf.math.l2_normalize(student_dvec, axis=-1)
        cosine_sim = tf.reduce_sum(teacher_dvec * student_dvec, axis=-1)
        loss = 1 - tf.reduce_mean(cosine_sim)

        self.loss_tracker.update_state(loss)
        self.sim_tracker.update_state(cosine_sim)

        return {"loss": self.loss_tracker.result(),
                "cosine_sim": self.sim_tracker.result()}

input_shape = (40, 40, 1)
student_model = build_student_model(input_shape)
student_model.summary()
distiller = DVectorDistiller(student_model, teacher_dvec_model)
distiller.compile(optimizer=tf.keras.optimizers.Adam(1e-4))

callbacks = [
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", factor=0.5, patience=10, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=20, restore_best_weights=True
    )
]

history = distiller.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    batch_size=32,
    epochs=700,
    callbacks=callbacks
)

model_name="dvector_student_model"

student_model.save(f"{model_name}.h5")
def convert_to_tflite(keras_model, output_filename):
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    tflite_model = converter.convert()

    with open(os.path.join('models', output_filename), 'wb') as f:
        f.write(tflite_model)
!mkdir models/
convert_to_tflite(student_model, f'{model_name}.tflite')

test_results = distiller.evaluate(X_test, y_test)
print(f"Test Loss: {test_results[0]:.4f}, Test Similarity: {test_results[1]:.4f}")

def compare_dvectors(teacher, student, X):
    teacher_dvec = teacher.predict(X)
    student_dvec = student.predict(X)

    teacher_dvec = tf.math.l2_normalize(teacher_dvec, axis=-1)
    student_dvec = tf.math.l2_normalize(student_dvec, axis=-1)

    cosine_sim = tf.reduce_mean(tf.keras.losses.cosine_similarity(
        teacher_dvec, student_dvec, axis=-1
    ))
    print(f"Mean D-vector Cosine Similarity: {cosine_sim.numpy():.4f}")

    mse = tf.reduce_mean(tf.square(teacher_dvec - student_dvec))
    print(f"Mean Squared Error: {mse.numpy():.4f}")

compare_dvectors(teacher_dvec_model, student_model, X_test[:1000])
