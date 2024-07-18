import bentoml
from bentoml.io import JSON


# retrieve model 
model_ref = bentoml.xgboost.get("poleng_xgb_binary:latest")
dv = model_ref.custom_objects["dict_vectorizer"]
nominal_vars = model_ref.custom_objects["nominal_variables"]

# initialize runner
model_runner = model_ref.to_runner()

# set up service
svc = bentoml.Service("political_engagement_classifier", runners=[model_runner])

@svc.api(input=JSON(), output=JSON())

async def classify(participant_profile):
    vector = dv.transform(participant_profile)
    prediction = await model_runner.predict.async_run(vector)

    print(f"----------------{prediction}-----------------")

    result = prediction[0]

    if result > 0.799:
        return {"message": "This person would benefit from political training."}
    else:
        return {"message": "No intervention required."}