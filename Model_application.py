# Завантаження моделі
autoencoder = load_model(model_path)

# Завантаження зображення
def preprocess_image(image):
    image = image.astype('float32') / 255.
    image = np.clip(image, 0., 1.)
    return np.reshape(image, (1, 32, 32, 3))

# Приклад використання з тестового набору
noisy_image = preprocess_image(x_test_noisy[0])
denoised_image = autoencoder.predict(noisy_image)

# Візуалізація результатів
plt.figure(figsize=(10, 2))

# Вихідне зображення зі шумом
ax = plt.subplot(1, 2, 1)
plt.imshow(noisy_image.reshape(32, 32, 3))
plt.title("Noisy Image")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

# Відновлене зображення
ax = plt.subplot(1, 2, 2)
plt.imshow(denoised_image.reshape(32, 32, 3))
plt.title("Denoised Image")
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

plt.show()
