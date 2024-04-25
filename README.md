# AI-Python-Demo

build AI demo for TribeTripe to check

To run the program, you need the following steps:

# Before everything

you should make sure you environment is good to run python
and you have already make the python installed

# Install packages in ChatBot.py

```python
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
```

you can run following commands to install them:

```bash
pip install pinecone
pip install openai
pip install dotenv
```

# Create your own .env file

it only need 2 env vars:

```txt
OPEN_AI_TOKEN=
PINECONE_TOKEN=
```

To get the token you need to apply from the public website of pinecone and openai:

1. [pinecone website](https://www.pinecone.io/)
2. [openai website](https://openai.com/)

# For beginner

If it's your first time to run the code,
you need to change main to like following:

```python
if __name__ == "__main__":
    main(True)
```

It will training the data to vectors and store them in pinecone database

If you are not the first time runner:

keep the code like that, or you will have duplicated vectors in database(output will be bad)

```python
if __name__ == "__main__":
    main(True)
```

# Start the program

run following command in your terminal:

```bash
python ChatBot.py
```

# Price

pinecone's free tier can help use run the program
But we need about 2.02 USD per 1M tokens(1M in and 1M out) to use openai's functions
