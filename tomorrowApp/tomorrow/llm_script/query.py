"""
based on https://github.com/togethercomputer/together-cookbook/blob/main/Text_RAG.ipynb
"""
import os
import json
from dotenv import load_dotenv
from typing import List
import numpy as np
import requests
from sklearn.metrics.pairwise import cosine_similarity

from django.conf import settings

# messages for the chat
chat_messages = []
TL_data = []


def load_json(path: str = 'data/tomorrowland.json'):
    """
    load the json file

    Args:
        path (str): path to file
    """

    # make sure to use the global TL_data
    global TL_data

    with open(path, 'r') as file:
        TL_data = json.load(file)


def llm(system_query: str, query: str, token_key: str, model: str) -> str:
    """
    code from: https://medium.com/@yashpaddalwar/how-to-access-free-open-source-llms-like-llama-3-from-hugging-face-using-python-api-step-by-step-5da80c98f4e3

    Args:
        system_query (str): system message for the prompt
        query (str): user query
        token_key (str): token key for the API
        model (str): model to run the LLM for

    Returns:
        str: string response of the LLM
    """

    url = "https://huggingface.co/api/inference-proxy/together/v1/chat/completions"

    # post parameters
    parameters = {
        "max_new_tokens": 5000,
        "temperature": 0.01,
        "top_k": 50,
        "top_p": 0.95,
        "return_full_text": False,
        "do_sample": False
    }

    # post headers (including token)
    headers = {
        'Authorization': f'Bearer {token_key}',
        'Content-Type': 'application/json'
    }

    messages = [
        {"role": "system", "content": system_query},
        {"role": "user", "content": query},
    ]

    # post payload (model + input)
    payload = {
        "model": model,
        "messages": messages,
        "parameters": parameters
    }

    # post the request with given headers and payload and await result
    response = requests.post(url, headers=headers, json=payload)
    try:
        response.json()["error"]
        return response.json()
    except KeyError:
        response_text = response.json()["choices"][0]["message"]["content"].strip()

        return response_text
    except requests.exceptions.JSONDecodeError:
        print(response)
        return ""


def generate_embeddings(input_texts: List[str], model: str, token_key: str) -> List[List[float]]:
    """
    Generate embeddings using the Hugging Face API

    Args:
        input_texts (List[str]): a list of string input texts.
        model (str): An API string for a specific embedding model of your choice.
        token_key (str): Token key for the api.

    Returns:
        List[List[float]]: a list of embeddings. Each element corresponds to the each input text.
    """

    url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model}"

    # post header (including token)
    headers = {
        'Authorization': f'Bearer {token_key}',
        'Content-Type': 'application/json'
    }

    # post payload (model + input)
    payload = {
        "model": model,
        "inputs": input_texts
    }

    response = requests.post(url, headers=headers, json=payload)

    return response.json()


def retreive(query: str, top_k: int = 5, index: np.ndarray = None, token_key: str = "") -> List[int]:
    """
    Retrieve the top-k most similar items from an index based on a query.

    Args:
        query (str): The query string to search for.
        top_k (int): The number of top similar items to retrieve. Defaults to 5.
        index (np.ndarray): The index array containing embeddings to search against. Defaults to None.
        token_key (str): The token key needed for the API.
    Returns:
        List[int]: A list of indices corresponding to the top-k most similar items in the index.
    """

    # create embeddings and find similarities
    query_embedding = generate_embeddings([query], 'BAAI/bge-base-en-v1.5', token_key)
    try:
        query_embedding = query_embedding[0]
    except IndexError:
        print(query_embedding)
        return [0 for _ in range(top_k)]
    similarity_scores = cosine_similarity([query_embedding], index)

    return np.argsort(-similarity_scores)[0][:top_k]


def generate(messages: List[dict]) -> List[dict]:
    """
    generate the text

    Args:
        messages (List[dict]): previous messages.

    Returns:
        List[dict]: all current messages
    """

    # set model and token
    model = "deepseek-ai/DeepSeek-R1"
    token = os.getenv("huggingface_key")

    # check if last message was from a user
    if messages[-1]["role"] != "user":
        return messages.append({"role": "system", "content": "You haven't asked/said anything yet"})

    # get the query
    query = messages[-1]["content"]

    # get a hidden query which better matches the format of the data
    system_query = "You're a system trying to get more information from a user prompt to try and gather any information about the following: {'name': '', 'stage': '', 'host': '', 'data': '', 'year': '', 'weekday': '', 'weekend': '', 'genre': '', 'time_start': '', 'time_end': ''}.\nAn example of a filled dict is: `{'name': 'DJ Mars', 'stage': 'The Gathering', 'host': 'The Gathering hosted by MC Gunner', 'data': '18 July', 'year': '2024', 'weekday': 'Thursday', 'weekend': 'Weekend 1', 'genre': '', 'time_start': '13:30 ', 'time_end': ' 14:15'}`, answer in the form of a filled dict based on the user query only the information you know from the user prompt, all other values stay as empty strings. If the query is about time then estimate the time the user will need based on an average.\n\n[Only give the dict you have filled in not the example the system example]"
    mid_query = llm(system_query, query, token, model)
    try:
        mid_query = mid_query.split("</think>")[1].strip()
    except IndexError:
        print(mid_query.strip())
        return messages.append({"role": "system", "content": "Something went wrong in the mid_query (llm_scipt/query.py @ line 164)"})

    # make sure the data is loaded
    if len(TL_data) == 0:
        load_json()

    # create an embedding list
    to_embed = []
    for artist in TL_data:
        to_embed.append(str(artist).strip())

    # generate retrieve the relevant data
    embeddings = generate_embeddings(to_embed, 'BAAI/bge-base-en-v1.5', token)
    indices = retreive(mid_query, 20, embeddings, token)
    TL_info = [TL_data[index] for index in indices]

    # set the answering query and generate the answer
    system_query = "You know anything about the performances at Tomorrowland just from the information given. Format your answer as HTML with bootstrap 4.3.1 classes not as markdown esque"
    query = f"{query}. Using the following information {TL_info}"

    result = llm(system_query, query, token, model)

    try:
        result = result.split("</think>")[1].strip()
        return messages.append({"role": "system", "content": result})
    except IndexError:
        print(result.strip())
        return messages.append({"role": "system", "content": "Something went wrong in the final message (llm_scipt/query.py @ line 192)"})


def main():
    url = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    url = "deepseek-ai/DeepSeek-R1"
    token = os.getenv("huggingface_key")

    system_query = "You're a system trying to get more information from a user prompt to try and gather any information about the following: {'name': '', 'stage': '', 'host': '', 'data': '', 'year': '', 'weekday': '', 'weekend': '', 'genre': '', 'time_start': '', 'time_end': ''}.\nAn example of a filled dict is: `{'name': 'DJ Mars', 'stage': 'The Gathering', 'host': 'The Gathering hosted by MC Gunner', 'data': '18 July', 'year': '2024', 'weekday': 'Thursday', 'weekend': 'Weekend 1', 'genre': '', 'time_start': '13:30 ', 'time_end': ' 14:15'}`, answer in the form of a filled dict based on the user query only the information you know from the user prompt, all other values stay as empty strings. If the query is about time then estimate the time the user will need based on an average.\n\n[Only give the dict you have filled in not the example the system example]"
    query = "Just arrived at the venue (15:00), what are the artists that start soon?"
    query = "Just finished watching Nina Black on the The Gathering stage, gonna eat dinner and after that what artists can I join in with without joining in the middle of their performance?"

    mid_query = llm(system_query, query, token, url)

    mid_query = mid_query.split("</think>")[1].strip()

    print(mid_query)

    with open('./data/tomorrowland.json', 'r') as file:
        TL_data = json.load(file)

    # Concatenate the title, overview, and tagline of each movie
    # this makes the text that will be embedded for each movie more informative
    # as a result the embeddings will be richer and capture this information.

    to_embed = []
    for artist in TL_data:
        to_embed.append(str(artist).strip())

    embeddings = generate_embeddings(to_embed, 'BAAI/bge-base-en-v1.5', token)
    # query_embedding = generate_embeddings([mid_query], 'BAAI/bge-base-en-v1.5', token)[0]

    # similarity_scores = cosine_similarity([query_embedding], embeddings)

    # indices = np.argsort(-similarity_scores)[0]

    # i = 0

    # for index in indices:
    #     artist = TL_data[index]
    #     artist_str = f"{artist['name']} (on {artist['stage']}) | {artist["time_start"]}-{artist["time_end"]} |"
    #     print(f"index {i:>2.0f}: {artist_str} - {similarity_scores[0][index]}")

    #     i += 1

    indices = retreive(mid_query, 20, embeddings, token)
    TL_info = [TL_data[index] for index in indices]

    system_query = "You know anything about the performances at Tomorrowland just from the information given. Format your answer as HTML not as markdown esque"
    query = f"{query}. Using the following information {TL_info}"

    print(llm(system_query, query, token, url))


if __name__ == '__main__':
    load_dotenv(dotenv_path=".env")
    main()
