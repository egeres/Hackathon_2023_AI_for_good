import os

from deepface import DeepFace

def get_features(img_path):
    # Deepface
    df = DeepFace.analyze(img_path=img_path, actions=['age', 'gender', 'race', 'emotion'])
    return df

def get_features_batch(dir):
    df = []
    for file in os.listdir(dir):
        df.extend(get_features(os.path.join(dir, file)))
    return df



if __name__ == '__main__':
    df = get_features_batch("data/")
