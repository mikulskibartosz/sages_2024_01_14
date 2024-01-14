from prophet.serialize import model_from_json
import bentoml

model_path = ".../data/models/prophet/model.json" # TODO zmienić ścieżkę!!!

with open(model_path, 'r') as fin:
    model_json = fin.read()

model = model_from_json(model_json)

bentoml.picklable_model.save_model(
    "SagesModel",
    model,
    signatures={"predict": {"batchable": False}}
)