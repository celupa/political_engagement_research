import requests
import json


# use this script to test the aws lambda function 
# classifying the political engagement of an individual 
api_url = "https://irc7lzpuuh.execute-api.eu-north-1.amazonaws.com/political_engagement_classifier/poleng_classifier"
headers = {"Content-Type": "application/json"}

# payloads
# note: variable defintiions available in "misc/post_variable_definitions.json"
# the profile below represents a politically disengaged, native zimbabwean man having completed the questionnaire
# in a computer assisted setting (interviewer present) and who has been reported to be
# interested in the exchange. His data indicates that he is single, young (gen y) 
# and living in a mid-sized family. He belong to the middle-class and is a manual laborer
apolitical = {"country": "716",
                "mode": "1",
                "settlement": "5",
                "respint": 1,
                "intprivacy": "1",
                "sex": "1",
                "immigrant": "1",
                "immigrant_mother": "1",
                "immigrant_father": "1",
                "birth_country": "1",
                "birth_country_mother": "1",
                "birth_country_father": "1",
                "citizenship": "1",
                "household_size": 5,
                "lives_with_parents": "2",
                "marital_status": "6",
                "children_number": 0,
                "education": 4,
                "education_spouse": 0,
                "education_mother": 1,
                "employment": "6",
                "employment_spouse": "0",
                "profession": "1",
                "profession_spouse": "0",
                "profession_father": "8",
                "employment_sector": "0",
                "chief_earner": "2",
                "savings": "1",
                "subjective_social_class": 3,
                "income_scale": 2,
                "religion": "2",
                "generation": 4}

# this user profile speaks for an older (silent generation), politically engaged, 
# australian woman that has completed and sent the survey by mail/post.
# somewhat interested by the study, this lady reported to be married, having an income
# in the lower bracket and having worked as a manual laborer
political = {"country": "36",
            "mode": "4",
            "settlement": "1",
            "respint": 2,
            "intprivacy": "1",
            "sex": "2",
            "immigrant": "1",
            "immigrant_mother": "2",
            "immigrant_father": "2",
            "birth_country": "0",
            "birth_country_mother": "0",
            "birth_country_father": "0",
            "citizenship": "1",
            "household_size": 2,
            "lives_with_parents": "1",
            "marital_status": "1",
            "children_number": 4,
            "education": 4,
            "education_spouse": 2,
            "education_mother": 2,
            "employment": "5",
            "employment_spouse": "1",
            "profession": "6",
            "profession_spouse": "8",
            "profession_father": "9",
            "employment_sector": "3",
            "chief_earner": "2",
            "savings": "4",
            "subjective_social_class": 5,
            "income_scale": 1,
            "religion": "1",
            "generation": 1}

# send post request
response = requests.post(api_url, headers=headers, data=json.dumps(apolitical))

# print status and response
print(f"Status Code: {response.status_code}")
print(f"Response Body: {response.json()}")




