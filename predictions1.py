def predict_image(path, model):
    import cv2
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (28,28))/255
    y = model.predict_on_batch(img.reshape(1,28,28,1)).argmax()
    return y
