import os
import json
import uuid
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
data_path = os.path.join(os.path.dirname(__file__), "./data/data.json")
IndexName = "triptribe-vectors"

pc = Pinecone(api_key=os.getenv("PINECONE_TOKEN"))
client = OpenAI(api_key = os.getenv("OPEN_AI_TOKEN"))

def CheckIndex(indexName:str):
    if any(index['name'] == indexName for index in pc.list_indexes()):
        return True
    else:
        pc.create_index(name=indexName, dimension=1536, spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        ))
        print(indexName + " index Created")
        return False

def GetIndex(indexName:str):
    CheckIndex(indexName)
    return pc.Index(indexName)

def Json2Txt(filepath:str):
    with open(filepath, 'r') as f:
        data = json.load(f)
    return [json.dumps(record) for record in data]

def Embedding(data, model="text-embedding-3-small"):
    if isinstance(data, str):
        data = [data]
    embedding = client.embeddings.create(input=data, model=model)
    if hasattr(embedding, 'data'):
        embedding_list = [record.embedding for record in embedding.data]
    else:
        embedding_list = []
    return embedding_list

def GetEmbeddingList(dataPath:str):
    data = Json2Txt(dataPath)
    return Embedding(data)

def InsertData(indexName:str, dataPath:str):
    embedding_list = GetEmbeddingList(dataPath)
    lineList = Json2Txt(dataPath)
    index = GetIndex(indexName)
    metaList = [{"text": line} for line in lineList]
    idList = [str(uuid.uuid4()) for _ in range(len(embedding_list))]
    result = list(zip(idList, embedding_list, metaList))
    index.upsert(vectors=result)
    print("Data Inserted")

def GetDataFromPinecone(indexName:str, question:str):
    index = GetIndex(indexName)
    EmbeddedQ = Embedding(question)
    result = index.query(vector=[EmbeddedQ], top_k=10, include_metadata=True)
    rf1 = result.matches[0].metadata['text']
    rf2 = result.matches[1].metadata['text']
    rf = rf1 + rf2
    print (rf)
    return rf

def ChatBot(questions:str):
    complation = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "All REFERENCE as the answer straightly strictly!!!!\
                    You are a helpful assistant for giving recommend restaurants for users and you can only use json format to answer the questions. \
                    The answer format following the following structure: \
                    {\"reason\": \"The reason why you choose the places.\" \"place_list\"(You can only use the data in REFERENCE and cannot adapt it yourself. If the reference is not empty, this list cannot be empty either!!!): a list contains the restaurants in REFERENCE strictly following the origin restaurant data structure you get, \"introduction\": give some beautiful introduction for the city\}, useREFERENCE as answer straightly strictly;\
                    If the question and REFERENCE do not seem to match, you do not need to recommend, but rather persuade the user to provide a description that better meets your requirements, but you should not mention that it is caused by too little data, but ask \
                    the user to give more description."
            },
            {
                "role": "user",
                "content": questions
            },
        ],
        temperature=0.7,
    )
    return complation.choices[0].message.content

def main(IsTrain:bool=False):
    CheckIndex(IndexName)
    if IsTrain:
        InsertData(IndexName, data_path)
    while True:
        question = input("Please input your question: ")
        info = GetDataFromPinecone(IndexName, question)
        question = f"""
        REFERENCE: {info}
        QUESTION: {question}
        """
        if question == "exit":
            break
        else:
            response = ChatBot(question)
            stopsignal = '='*200
            print(f"answer: {response}")
            print("\n" + stopsignal + "\n")

if __name__ == "__main__":
    main()
