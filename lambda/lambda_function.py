import pickle
import xgboost as xgb


# get model
with open("poleng_xgb.bin", "rb") as fin:
    dv, model = pickle.load(fin)


def lambda_handler(participant_profile, context):
    vector = dv.transform(participant_profile)
    xgb_vector = xgb.DMatrix(vector)
    prediction = model.predict(xgb_vector)

    result = prediction[0]

    if result > 0.799:
        return {"message": "This person would benefit from political training."}
    else:
        return {"message": "No intervention required."}
    
