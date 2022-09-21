from scipy.special import softmax
import tensorflow as tf
import numpy as np
import classification_pb2 as pb
import classification_pb2_grpc as pbg
import grpc
from concurrent import futures
from PIL import Image
from io import BytesIO

class_names = ['apple_pie',
               'baklava',
               'beef_carpaccio',
               'beignets',
               'bread_pudding',
               'breakfast_burrito',
               'caesar_salad',
               'caprese_salad',
               'carrot_cake',
               'cheesecake',
               'chicken_curry',
               'chicken_wings',
               'chocolate_cake',
               'churros',
               'clam_chowder',
               'club_sandwich',
               'creme_brulee',
               'deviled_eggs',
               'donuts',
               'eggs_benedict',
               'falafel',
               'filet_mignon',
               'foie_gras',
               'french_fries',
               'french_toast',
               'fried_calamari',
               'fried_rice',
               'garlic_bread',
               'greek_salad',
               'grilled_salmon',
               'guacamole',
               'hamburger',
               'hot_dog',
               'huevos_rancheros',
               'ice_cream',
               'lasagna',
               'lobster_bisque',
               'macaroni_and_cheese',
               'macarons',
               'miso_soup',
               'nachos',
               'omelette',
               'onion_rings',
               'oysters',
               'paella',
               'pancakes',
               'panna_cotta',
               'peking_duck',
               'pizza',
               'pork_chop',
               'poutine',
               'prime_rib',
               'pulled_pork_sandwich',
               'ramen',
               'ravioli',
               'red_velvet_cake',
               'risotto',
               'samosa',
               'sashimi',
               'scallops',
               'shrimp_and_grits',
               'spaghetti_bolognese',
               'spaghetti_carbonara',
               'spring_rolls',
               'steak',
               'strawberry_shortcake',
               'sushi',
               'tacos',
               'takoyaki',
               'tiramisu',
               'waffles']


class Classifier:
    def __init__(self) -> None:
        interpreter = tf.lite.Interpreter(model_path='model.tflite')
        self.model = interpreter.get_signature_runner('serving_default')
        print(interpreter.get_input_details())

    def predict(self, img):
        p = self.model(input_6=img)['outputs']
        # return np.argmax(softmax(p))
        s = softmax(p)
        id = np.argmax(s)
        return id, s[0][id]


class ClassImgServicer(pbg.ClassImgServiceServicer):
    def __init__(self) -> None:
        self.classifier = Classifier()

    def ClassImg(self, request, context):
        img = Image.open(BytesIO(request.img))
        img = img.resize((254, 254))
        img = np.asarray(img, 'float32')
        img = np.array([img])
        id, ac = self.classifier.predict(img)
        print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[id], 100 * ac)
        )
        return pb.ResponseClassImg(id=id)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    pbg.add_ClassImgServiceServicer_to_server(
        ClassImgServicer(), server)
    server.add_insecure_port('[::]:4005')
    print("Server started at 50051")
    server.start()
    server.wait_for_termination()


serve()
