# askRobin-TomorrowlandApp
Django App for technical assessment of askRobin it.
The Tomorrowland data is retrieved from the [festival viewer](https://festivalviewer.com/tomorrowland/lineup/2024) website. From this website the data has been scraped using the `tomorrowland_dataparse.js`
The LLM used for text generation is [deepseek-ai/DeepSeek-R1](https://huggingface.co/deepseek-ai/DeepSeek-R1) and the embedding model that was used is [BAAI/bge-base-en-v1.5](https://huggingface.co/BAAI/bge-base-en-v1.5)
In `query.py` are the functions related to the text generation.
The method applied is an altered version of this [cookbook](https://github.com/togethercomputer/together-cookbook/blob/main/Text_RAG.ipynb). To make the query more in line with the to be embedded data an extra LLM layer has been added to extract the relevant info from the user prompt to better compare with the data. This yields results with higher similarity and correctness with respect to the user prompt.