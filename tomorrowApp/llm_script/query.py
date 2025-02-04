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


chat_messages = []


def llm(system_query: str, query: str, token_key: str, model: str) -> str:
    """
    code from: https://medium.com/@yashpaddalwar/how-to-access-free-open-source-llms-like-llama-3-from-hugging-face-using-python-api-step-by-step-5da80c98f4e3
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
        return ""


def generate_embeddings(input_texts: List[str], model: str, token_key: str) -> List[List[float]]:
    """Generate embeddings from Together python library.

    Args:
        input_texts: a list of string input texts.
        model_api_string: str. An API string for a specific embedding model of your choice.

    Returns:
        embeddings_list: a list of embeddings. Each element corresponds to the each input text.
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
        top_k (int, optional): The number of top similar items to retrieve. Defaults to 5.
        index (np.ndarray, optional): The index array containing embeddings to search against. Defaults to None.
    Returns:
        List[int]: A list of indices corresponding to the top-k most similar items in the index.
    """

    # create embeddings and find similarities
    query_embedding = generate_embeddings([query], 'BAAI/bge-base-en-v1.5', token_key)[0]
    similarity_scores = cosine_similarity([query_embedding], index)

    return np.argsort(-similarity_scores)[0][:top_k]


def generate(messages: List[dict]) -> List[dict]:
    model = "deepseek-ai/DeepSeek-R1"
    token = os.getenv("huggingface_key")

    if messages[-1]["role"] != "user":
        return messages.append({"role": "system", "content": "You haven't asked/said anything yet"})

    query = messages[-1]["content"]

    system_query = "You're a system trying to get more information from a user prompt to try and gather any information about the following: {'name': '', 'stage': '', 'host': '', 'data': '', 'year': '', 'weekday': '', 'weekend': '', 'genre': '', 'time_start': '', 'time_end': ''}.\nAn example of a filled dict is: `{'name': 'DJ Mars', 'stage': 'The Gathering', 'host': 'The Gathering hosted by MC Gunner', 'data': '18 July', 'year': '2024', 'weekday': 'Thursday', 'weekend': 'Weekend 1', 'genre': '', 'time_start': '13:30 ', 'time_end': ' 14:15'}`, answer in the form of a filled dict based on the user query only the information you know from the user prompt, all other values stay as empty strings. If the query is about time then estimate the time the user will need based on an average.\n\n[Only give the dict you have filled in not the example the system example]"

    mid_query = llm(system_query, query, token, model)

    mid_query = mid_query.split("</think>")[1].strip()

    with open('./data/tomorrowland.json', 'r') as file:
        TL_data = json.load(file)

    to_embed = []
    for artist in TL_data:
        to_embed.append(str(artist).strip())

    embeddings = generate_embeddings(to_embed, 'BAAI/bge-base-en-v1.5', token)

    indices = retreive(mid_query, 20, embeddings, token)
    TL_info = [TL_data[index] for index in indices]

    system_query = "You know anything about the performances at Tomorrowland just from the information given. Format your answer as HTML not as markdown esque"
    query = f"{query}. Using the following information {TL_info}"

    return messages.append({"role": "system", "content": llm(system_query, query, token, model).split("</think>")[1].strip()})


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