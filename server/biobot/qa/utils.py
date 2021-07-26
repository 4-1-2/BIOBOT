import random
def get_suggested_question(plant_name: str, disease: str):
    """
        Suggest initial question based on the status of the plant
    """
    #plant_name = ' '.join(plant_name.split('_'))
    plant_name = plant_name.strip().lower()
    #disease = ' '.join(disease.split('_'))  # clean subguions _
    disease = disease.strip().lower()
    if disease != 'healthy':
        # Questions related to agronomy
        questions = [
            'What is {}?'.format(disease),
            'What ecofriendly solutions exists to manage a {} plant with {} disease?'.format(plant_name, disease)
        ]
    else:
        questions = [
            'How can I enrich the soil for cultivation in the long term?',
            'Which are common diseases for {} plants?'.format(plant_name),
            'What do means biological control?',
            ''
        ]
    return random.choice(questions)

