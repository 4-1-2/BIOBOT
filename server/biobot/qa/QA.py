import os
import re
import openai

class OpenAIPlayGround:
    def __init__(
        self,
        path_credentials,
        temp=0.0, 
        engine='davinci', 
        max_tokens=100, 
        top_p=0.59, 
        frequency_pen=0, 
        presence_pen=0,
        stop=["\n\nQ:", "\nA:"]
    ):
        assert temp >= 0.0 and temp < 1.0, 'temperature must be between 0 and 1'
        assert engine in ['davinci', 'curie', 'babbage', 'ada'], 'The engine is incorrect'
        self.temperature = temp
        self.engine = engine
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.frequency_pen = frequency_pen
        self.presence_pen = presence_pen
        self.base_prompt = self._get_base_prompt()
        self.stop = stop
        self._give_credentials(path_credentials)

    def __call__(self, comming_text: str, prev_text: str=''):
        comming_text = comming_text.strip()
        query_prompt, acumm_chat = self._append_text(comming_text, prev_text)
        response = openai.Completion.create(
            engine=self.engine,
            prompt=query_prompt,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p,
            frequency_penalty=self.frequency_pen,
            presence_penalty=self.presence_pen,
            stop=["\n\nQ:", "\nA:"]
        )
        nchoices = len(response['choices'])
        text = response['choices'][0]['text'].strip() if nchoices > 0 else ''

        acumm_chat = '{} {}'.format(acumm_chat, text)
        return {
            'chat_acumm': acumm_chat,
            'answer': text,
            'nchoices': nchoices
        }
    
    def _append_text(self, new_text: str, prev_text):
        searchQ = re.search(r'.*(Q:)(.*)', new_text)
        if searchQ:
            question = searchQ.groups()[1].strip()
        else:
            question = new_text.strip()
        question = 'Q: {}\nA:'.format(question)
        prev_text = prev_text.strip()
        text_chat = '{}\n\n{}'.format(prev_text, question)
        return self.base_prompt + text_chat, text_chat

    def _give_credentials(self, path_credentials):
        with open(path_credentials, 'r') as fp:
            apikey = fp.readlines()
            apikey = ''.join(apikey).strip()
            openai.api_key = apikey

    def _get_base_prompt(self):
        return \
        """I am a highly intelligent farmer, I worked in agronomy industry for more than 30 years. I you ask me for recommendations or solutions widely used in the market about how to handle issues related to plant diseases, I will give you the answer. If you ask me a question that is nonsense, trickery, has no clear answer or is not related to agronomy topic, I will respond with "Unknown". I  have preferences for ecological solutions, but if there is none, I will give you the most common solution.

agronomy related topics: plant, soil, green, fungicide

Here are some examples how I will answer you:

Q: Which are common diseases for apple threes?
A: apple scab, cedar apple rust, bitter rock.

Q: What is the most accepted treatment for apple scab?
A: Rake up leaves and remove them from the orchard before May. Remove abandoned apple trees within 100 yards of your orchard. The University of Maine Cooperative Extension recommends applying preventive sprays such as Captan, sulfur, or other fungicides.

Q: When was the independence of the United States?
A: Not related to agronomy.

Q: Tell me some signs or evidence about fungal disease in my plant.
A:  When you look at powdery mildew on a lilac leaf, you’re actually looking at the parasitic fungal disease organism itself (Microsphaera alni). Bacterial canker of stone fruits causes gummosis, a bacterial exudate emerging from the cankers. The thick, liquid exudate is primarily composed of bacteria and is a sign of the disease, although the canker itself is composed of plant tissue and is a symptom.

Q: What is the best car in the world?
A: Not related to agronomy

Q: What are common diseases in humans?
A: Not related to agronomy

Q: Can you tell me your opinion about the new president?
A: Not related to agronomy"""

if __name__ == '__main__':
    api = OpenAIPlayGround() # using default parameters
    response = api(comming_text="Tell me about fungus diseases in plants")
    print(response['chat_acumm'])
    print(response['answer'])
    print(response['nchoices'])