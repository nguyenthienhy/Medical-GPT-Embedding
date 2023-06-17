

import openai
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import ast


# Thiết lập thông tin xác thực API
openai.api_key = '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'

# Đọc và xử lý bộ vector nhúng
def read_csv(filename):
    df = pd.read_csv('medical-embeddings.csv')
    embeddings = np.array(df['ada_v2'])
    sentences = df['CONTEXT'].tolist()
    return embeddings, sentences

# Tìm kiếm vector nhúng gần nhất
def find_nearest_embedding(question_embedding, embeddings, sentences):
    question_embedding = question_embedding.reshape(1,-1)
    similarity_scores = []
    for embedding in embeddings:
        similarity_scores.append(cosine_similarity(question_embedding, embedding)[0][0])
    nearest_index = np.argmax(similarity_scores)
    nearest_sentence = sentences[nearest_index]
    return nearest_sentence

def embed_question_with_api(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

# Hàm gửi câu hỏi và nhận câu trả lời từ GPT-3
def ask_question(question):
    response = openai.Completion.create( 
        model='text-davinci-003',
        prompt=question
    )
    if response.choices:
        return response.choices[0].text.strip()
    else:
        return "sorry, i can not answer."


# Hàm tương tác với người dùng
def chatbot_interaction():

    embeddings, sentences = read_csv('medical-embeddings.csv')

    embeddings_reformat = []

    for embedding_temp in embeddings:
        embedding_temp = np.fromstring(embedding_temp[1:-1], dtype=float, sep=',')
        embedding_temp = embedding_temp.reshape(1,-1)
        embeddings_reformat.append(embedding_temp)

    while True:

        user_input = input("Question: ")

        if user_input.lower() == 'bye':
            print("Chatbot: Goodbye!")
            break

        # Nhúng câu hỏi thành vector sử dụng API
        question_embedding = embed_question_with_api(user_input)
        
        if question_embedding is not None:

            # Chuyển đổi danh sách thành mảng numpy
            question_embedding = np.array(question_embedding)

            # print("question_embedding:", str(question_embedding.shape))

            try:

                answer = ask_question(user_input).replace("\n", " ")

                # print(answer)
                
                # Nhúng câu trả lời thành vector
                answer_embedding = embed_question_with_api(answer)
                
                if answer_embedding is not None:

                    # Chuyển đổi danh sách thành mảng numpy
                    answer_embedding = np.array(answer_embedding)

                    # print("answer_embedding:", str(answer_embedding.shape))

                    # Tìm vector nhúng gần nhất
                    nearest_sentence = find_nearest_embedding(answer_embedding, embeddings_reformat, sentences)

                    print("Chatbot:", nearest_sentence)

                else:
                    print("Chatbot: Sorry, I can't embed the answer.")
                
            except (ValueError, KeyError):
                print("Chatbot: Sorry, an error occurred while processing the response.")
        else:
            print("Chatbot: Sorry, I can't embed this question.")

# Gọi hàm tương tác với người dùng
chatbot_interaction()
