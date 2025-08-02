# from dotenv import load_dotenv
import os
import google.generativeai as genai
from prompts import HISTORY_QUESTION_PROMPT, TEST_PROMPT, get_essay_with_topic, HISTORY_ESSAY_TOPICS, HISTORY_ESSAY_TOPICS_2
import json
from mistralai import Mistral
import numpy as np
import time
from scipy.stats import norm
import pandas as pd


ENV_PATH = "./.env"
GEMINI_JSON = "./data/gemini.json"
MISTRAL_JSON = "./data/mistral.json"
GEMINI_TEST_JSON = "./data/gemini_test.json"
MISTRAL_TEST_JSON = "./data/mistral_test.json"
GEMINI_PROB = "./data/mistral_prob.json"
MISTRAL_PROB = "./data/gemini_prob.json"

MIN_VAL = 1e-5
# load_dotenv(ENV_PATH)
CURR_GEMINI_API = os.getenv("GEMINI_API_KEY_1")
# CURR_GEMINI_API = GEMINI_API_KEY_1
genai.configure(api_key=CURR_GEMINI_API)

gemini_model = genai.GenerativeModel(model_name="gemini-2.0-flash")
mistral_model = Mistral(api_key= os.getenv("MISTRAL_API_KEY"))


class Model():

    def init(self):
        pass
    def predict(self, text):
        gemini_prob, mistral_prob = load_prob()
        pred, value = predict_gemini_mistral(text, gemini_prob, mistral_prob)
        print(value)
        return pred, self.sigmoid(value)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

def get_gemini_output(prompt):

    response = gemini_model.generate_content(prompt)
    clean_response = remove_formatting(response.text.strip())
    json_output = json.loads(clean_response, strict=False)

    return json_output

def get_mistral_output(prompt):

    print(prompt)
    response = mistral_model.chat.complete(
        model = "mistral-large-latest",
        messages= [ {
            "role": "user",
            "content": prompt
            }
        ]
    )

    response_0 = response.choices[0].message.content
    clean = remove_formatting(response_0)
    json_output = json.loads(clean, strict=False)

    return json_output

def get_mistral_essay(topic, save_path = MISTRAL_JSON):
    dic = []

    with open (save_path) as f:
        dic = json.load(f)

    mistral_output = get_mistral_output(get_essay_with_topic(topic))
    mistral_output["topic"] = topic
    dic.append(mistral_output)

    # Processing and adding new essay
    with open(save_path, "w") as f:
        json.dump(dic, f)  

def remove_formatting(response):
    if (response.startswith("```json")):
        response = response[7:]
    if (response.endswith("```")):
        response = response[:-3]
    return response


# This function gets essays and updates the gemini.json file with new essays
def get_gemini_essay(topic, save_path = GEMINI_JSON):
    dic = []

    with open (save_path) as f:
        dic = json.load(f)

    gemini_output = get_gemini_output(get_essay_with_topic(topic))
    gemini_output["topic"] = topic
    dic.append(gemini_output)

    with open(save_path, "w") as f:
        json.dump(dic, f)

    return gemini_output

def get_topics(file_name):
    topics = []

    with open(file_name) as f:
        essays = json.load(f)

        for item in essays:

            try:
                topic = item["topic"]
                topics.append(topic)
            except:
                continue

    return topics

def clean_text(text):
    text = text.replace("\n", " ")
    punctuation_marks = [
        ".", ",", ";", ":", "?", "!","‘", "’", "“", "”", "(", ")", "[", "]", "{", "}", "<", ">",
        "-", "–", "—", "/", "\\", "|",
        "@", "#", "$", "%", "&", "*", "^", "_", "~", "=", "+"
    ]

    for item in punctuation_marks:
        text = text.replace(item, f" {item}")
    return text

def find_prob(file_name, write_path="", number_of_words=-1):

    total_words = 0
    model_dict = {}
    with open(file_name) as f:
        essays = json.load(f)
        all_str = ""
        for item in essays:
            all_str += f" {item["essay"]}"
        # all_str = all_str.lower()
        # words = all_str.split(" ")

        cleaned_text = clean_text(all_str.lower())
        arr = cleaned_text.split(" ")
        words = [a for a in arr if a != ""]

        words = words[:number_of_words]
        total_words = len(words)
        # print(total_words)
        for word in words:
            try:
                model_dict[word] += 1 / total_words
            except:
                model_dict[word] = 1 / total_words

        if write_path != "":
            with open(write_path, "w") as wf:
                json.dump(model_dict, wf)

        return model_dict, total_words

def find_total_words(file_name):

    with open(file_name) as f:
        essays = json.load(f)
        all_str = ""
        for item in essays:
            all_str += f" {item["essay"]}"
        all_str = all_str.lower()
        words = all_str.split(" ")
        return len(words)

def predict_gemini_mistral(text, gemini_dict: dict, mistral_dict: dict):

    text_arr = text.split(" ")

    total = 0
    for token in text_arr:
        Gi = MIN_VAL
        Mi = MIN_VAL
        try:
            Gi = gemini_dict[token]
        except:
            # continue
            pass
        try:
            Mi = mistral_dict[token]
        except:
            # continue
            pass
        # print(Gi, Mi)
        total += np.log(Gi / Mi)

    if total > 0:
        return 1, total
    else:
        return 0, total

def switch_api_key():
    global CURR_GEMINI_API
    if CURR_GEMINI_API == os.getenv("GEMINI_API_KEY_1"):
        CURR_GEMINI_API = os.getenv("GEMINI_API_KEY_2")
        genai.configure(api_key=CURR_GEMINI_API)

    else:
        CURR_GEMINI_API = os.getenv("GEMINI_API_KEY_1")
        genai.configure(api_key=CURR_GEMINI_API)

def add_essays(n, gemini_file_name, mistral_file_name, topics_list):

    topic = ""
    gemini_topics = get_topics(gemini_file_name)
    mistral_topics = get_topics(mistral_file_name)


    for i in range(n):
        for t in topics_list:
            if not (t in gemini_topics) and not (t in mistral_topics):
                topic = t
                gemini_topics.append(t)
                mistral_topics.append(t)
                break
        if topic == "":
            print("No more new topic!")
            break

        print(f"Staring topic {topic}")
        while True:
            try:
                gemini_output = get_gemini_essay(topic=topic, save_path=gemini_file_name)
                topic = gemini_output["topic"]
                break

            except Exception as e:
                print(e)
                switch_api_key()
                time.sleep(5 + norm.rvs(1, 1) * 5)
                continue

        while True:
            try:
                get_mistral_essay(topic=topic, save_path=mistral_file_name)
                break
            except Exception as e:
                print(e)
                time.sleep(20 + norm.rvs(1, 1) * 5)
        topic = ""

def load_prob(gemini_path=GEMINI_PROB, mistral_path=MISTRAL_PROB):
    gemini_dict = None
    mistral_dict = None

    with open(gemini_path) as f:
        gemini_dict = json.load(f)

    with open(mistral_path) as f:
        mistral_dict = json.load(f)

    return gemini_dict, mistral_dict

def fit(gemini_path=GEMINI_PROB, mistral_path=MISTRAL_PROB, num_of_words=-1):

    # Training files 
    find_prob(GEMINI_JSON, gemini_path, num_of_words)
    find_prob(MISTRAL_JSON, mistral_path, num_of_words) 

# gemini_path: Gemini path of the file to be evaluated on
# mistral_path: Mistral path of the file to be evaluated on
def find_results(gemini_path, mistral_path, pred_func):

    gemini_dict, mistral_dict = load_prob()
    # Gemini is 1, Mistral is 0, TP is FN is for Gemini, TN and FP is for Mistral
    TP = 0
    FN = 0
    TN = 0
    FP = 0

    with open(gemini_path) as f:
        json_inp = json.load(f)

        for item in json_inp:
            try:
                essay = item["essay"]
                if (pred_func(essay, gemini_dict, mistral_dict)[0]):
                    TP+=1
                else:
                    FN+=1
            except:
                continue

    with open(mistral_path) as f:
        json_inp = json.load(f)

        for item in json_inp:
            try:
                essay = item["essay"]
                if (pred_func(essay, gemini_dict, mistral_dict)[0]):
                    FP +=1
                else:
                    TN +=1
            except:
                continue
    
    acc = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1_score = 2 * precision * recall / (recall + precision)
    print(TP, TN, FP, FN)
    return acc, recall, precision, f1_score
        


if __name__ == "__main__":
    # gemini_topics = get_topics(GEMINI_TEST_JSON)
    # mistral_topics = get_topics(MISTRAL_TEST_JSON)
    # print(find_total_words(GEMINI_TEST_JSON))
    # print(find_total_words(MISTRAL_TEST_JSON))
    add_essays(300, GEMINI_JSON, MISTRAL_JSON, HISTORY_ESSAY_TOPICS_2)

    # fit()
    # result = find_results(GEMINI_TEST_JSON, MISTRAL_TEST_JSON, predict_gemini_mistral)
    # print(result)
    # lst = np.linspace(500, 250000)
    # df = pd.DataFrame(columns=["Accuracy", "Recall", "Precision", "f1_score"])
    # for num in lst:
    #     print(num)
    #     fit(num_of_words=int(num))
    #     result = find_results(GEMINI_TEST_JSON, MISTRAL_TEST_JSON, predict_gemini_mistral)
    #     print(len(df))
    #     df.loc[num] = result

    # print(df)
    # df.to_csv("./results.csv")



