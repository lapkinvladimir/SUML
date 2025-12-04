from app.predict import predict


def test_prediction():
    # пример произвольных валидных чисел для Iris
    result = predict([5.1, 3.5, 1.4, 0.2])

    # класс модели должен быть 0, 1 или 2
    assert result in [0, 1, 2]
