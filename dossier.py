# time constraints did not allow to completion of this todo list
# TODO: GENERAL
# implement PCA
# implement pipeline
# try unsupervised learning
# TODO: FORMAT ORIGINAL RESPONSE ARRAY
### implement transform_array()
## replaces alternative responses with null (except -3)
## smoothe "not applicable" (-3 > -1)
## birth_country to binary (1=native, 0=born in another country)
## rescale following vars to go from 1 to n:
# education
# profession
# profession_spouse
# profession_father
# religion
## streamline vars with -1:
# immigrant_mother
# immigrant_father
# education_spouse
# education_mother
# employment_spouse
# profession
# profession_spouse
# profession_father
# employment_sector
## descritizes continuous variables
# children number 4 = => 4
# household size 5 = => 5
# transforms vd 
## assign weights to array
# TODO: SUPPORT ANALYSES
# check correlations between variables and their weight
# assess model specification
# assess impact of weights on model


# below is a map of the features and their names
# key=questionnaire feature name, values=explicit feature name
COLUMNS_TO_RETAIN = {# methodological variables
                     "A_YEAR": "collection_year",
                     "B_COUNTRY": "country",                   
                     "Q_MODE": "mode",
                     "H_SETTLEMENT": "settlement",    
                     "K_DURATION": "interview_duration",
                     "E_RESPINT": "respint",
                     "F_INTPRIVACY": "intprivacy",
                     "W_WEIGHT": "weight",        
                     # demographic variables
                     "Q260": "sex",                            
                     "Q261": "birth_year",
                     "Q262": "age",
                     "Q263": "immigrant",
                     "Q264": "immigrant_mother",
                     "Q265": "immigrant_father",
                     "Q266": "birth_country",
                     "Q267": "birth_country_mother",
                     "Q268": "birth_country_father",
                     "Q269": "citizenship",
                     "Q270": "household_size",
                     "Q271": "lives_with_parents",
                     "Q273": "marital_status",
                     "Q274": "children_number",
                     "Q275": "education",
                     "Q276R": "education_spouse",
                     "Q277R": "education_mother",
                     "Q279": "employment",
                     "Q280": "employment_spouse",
                     "Q281": "profession",
                     "Q282": "profession_spouse",
                     "Q283": "profession_father",
                     "Q284": "employment_sector",
                     "Q285": "chief_earner",
                     "Q286": "savings",
                     "Q287": "subjective_social_class",
                     "Q288R": "income_scale",
                     "Q289": "religion",
                     # political variables (target)
                     "Q209": "signed_petitions",                 
                     "Q210": "joined_boycotts",
                     "Q211": "attended_peaceful_demonstrations",
                     "Q212": "joined_unofficial_strikes",
                     "Q213": "financed_campaigns",
                     "Q214": "contacted_government",
                     "Q215": "encouraged_political_action",
                     "Q216": "encouraged_voting",
                     "Q217": "stays_politically_updated",
                     "Q218": "signed_petitions_online",
                     "Q219": "encouraged_political_action_online",
                     "Q220": "organized_political_events",
                     "Q221": "voted_locally",
                     "Q222": "voted_nationally"}

