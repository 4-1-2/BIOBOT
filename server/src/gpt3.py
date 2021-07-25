import os
import openai

openai.api_key = ""

response = openai.Completion.create(
  engine="davinci",
  prompt="\n Q: What treatment should i do for Leaf scorch in tomato plant? \n A:",
  temperature=1,
  max_tokens=100,
  top_p=1,
  frequency_penalty=0.0,
  presence_penalty=0.0,
  stop=["\n"]
)


print(response )

#what treatment should i do for spider mites in tomato plant