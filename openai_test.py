import openai

openai.api_key = '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'

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

print(type(get_embedding("hello")))