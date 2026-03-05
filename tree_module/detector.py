from deepforest import main

class TreeDetector:

    def __init__(self, model_path=None):

        self.model = main.deepforest()

        self.model.load_model()

        if model_path:
            self.model.load_model(model_path)

    def detect(self, image_path):

        predictions = self.model.predict_image(path=image_path)

        return predictions