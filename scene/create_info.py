import json
import os

from simple_3dviz import TexturedMesh
from tqdm import tqdm

path = 'C:/Users/Flexatroid/Desktop/diploma/3D-FRONT-base/3D-FUTURE-model'
models = json.load(open("my_models_info.json"))
new_models = json.load(open("fixed_models.json"))

counter = 0
for model in tqdm(os.listdir(path)):
    try:
        counter += 1
        if model in new_models:
            continue

        model_path = os.path.join(path, model, "raw_model.obj")
        mesh = TexturedMesh.from_file(model_path)
        bbox = mesh.bbox
        sizes = bbox[1] - bbox[0]
        sizes = sizes.astype(float)
        new_models[model] = {
            "boundingBox": {
                "xLen": sizes[0],
                "yLen": sizes[1],
                "zLen": sizes[2]
            },
            "super-category": models[model]["super-category"],
            "category": models[model]["category"],
            "style": models[model]["style"]
        }
        if counter % 500 == 0:
            print("saved, ", counter)
            with open('fixed_models.json', 'w') as outfile:
                json.dump(new_models, outfile, indent=2)

    except Exception as e:
        print(model, " ERROR!")

print(len(new_models), len(models))

with open('fixed_models.json', 'w') as outfile:
    json.dump(new_models, outfile, indent=2)
