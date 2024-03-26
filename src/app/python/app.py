import fileinput
from flask import Flask, jsonify, request, current_app, make_response, abort
import json
import base64
import requests
from flask_cors import CORS, cross_origin
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
# from openai import OpenAI
from langchain_community.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.chains import LLMChain
from langchain.prompts import HumanMessagePromptTemplate
from langchain.prompts import SystemMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import UnstructuredEmailLoader
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.utilities.dalle_image_generator import DallEAPIWrapper
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.agents.agent_toolkits import GmailToolkit
from langchain_community.document_loaders.figma import FigmaFileLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.agents import AgentType, initialize_agent
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain.output_parsers import PydanticOutputParser
from mimetypes import guess_extension
from werkzeug.utils import secure_filename
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator
    )
from langchain_experimental.synthetic_data import (
    DatasetGenerator,
    create_data_generation_chain,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,)
import os
import openai
 
openai.api_key = os.environ["OPENAI_API_KEY"]
class AccountDetails(BaseModel):
# llm = OpenAI(temperature=0.9)
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
@app.route('/api/data')
def get_data():
 queryparam = request.args.get('query')
 queryparamCourse = request.args.get('course')
 if queryparamCourse == "Figma":
     loader = YoutubeLoader.from_youtube_url(
    "https://youtu.be/II-6dDzc-80?si=b2lJN8u4L9zh5saf", add_video_info=False
)
 elif queryparamCourse == "Sql":
     loader = YoutubeLoader.from_youtube_url(
    "https://youtu.be/HXV3zeQKqGY?si=MiAxAauoaI575BH0", add_video_info=False
)
 docs = loader.load()
 combined_docs = [doc.page_content for doc in docs]
 text = " ".join(combined_docs)
 text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
 splits = text_splitter.split_text(text)
 embeddings = OpenAIEmbeddings()
 vectordb = FAISS.from_texts(splits, embeddings)
 result_docs = vectordb.similarity_search(queryparam)
 result_docs_page_content = " ".join([d.page_content for d in result_docs])
 chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that answers questions about youtube videos based on the youtube video's transcript: {result_docs} If you dont find the answer from  youtube video's transcript, say I dont Know!"
            )
        ),
        HumanMessagePromptTemplate.from_template(queryparam),
    ]
)
 qa_chain = LLMChain(llm=OpenAI(), prompt=chat_template)
 response = qa_chain.run(question=queryparam, docs=result_docs_page_content)
 return jsonify(response)


@app.route('/api/email')
def get_email_data():
 queryparam = request.args.get('query') 
 loader = UnstructuredEmailLoader("/Users/pandiarajanr/Desktop/Angular/PR AI/genAIPR/src/assets/niyo.eml", mode="elements")
 docs = loader.load()
 combined_docs = [doc.page_content for doc in docs]
 text = " ".join(combined_docs)
 text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
 splits = text_splitter.split_text(text)
 embeddings = OpenAIEmbeddings()
 vectordb = FAISS.from_texts(splits, embeddings)
 result_docs = vectordb.similarity_search(queryparam)
 result_docs_page_content = " ".join([d.page_content for d in result_docs])
 chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "You are a helpful assistant that answers questions from {result_docs} If you dont find the answer, say I dont Know!"
            )
        ),
        HumanMessagePromptTemplate.from_template(queryparam),
    ]
)
 qa_chain = LLMChain(llm=OpenAI(), prompt=chat_template)
 print(queryparam , result_docs_page_content)
 response = qa_chain.run(question=queryparam, docs=result_docs)
 return jsonify(response)

@app.route('/api/gmail')
def get_gmail_data():
 queryparam = request.args.get('query') 
 toolkit = GmailToolkit()
 llm = ChatOpenAI(model="gpt-3.5-turbo", temperature= 0.0)
 agent = initialize_agent(tools=toolkit.get_tools(), llm=llm, verbose=True, max_iterations=1000, max_execution_time=1600, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION)
 response = agent.run(queryparam)
 return jsonify(response)

@app.route('/api/figma')
def get_figma_code():
 queryparam = request.args.get('query')
 figma_loader = FigmaFileLoader("Use your Figma Key")
 index = VectorstoreIndexCreator().from_loaders([figma_loader])
 figma_doc_retriever = index.vectorstore.as_retriever()
 system_prompt_template = """You are expert coder PR. Use the provided design context to create idiomatic HTML/CSS code as possible based on the user request.
    Everything must be inline in one file and your response must be directly renderable by the browser.
    Figma file nodes and metadata: {context}"""

 human_prompt_template = "Code the {text}. Ensure it's mobile responsive"
 system_message_prompt = SystemMessagePromptTemplate.from_template(
        system_prompt_template
    )
 human_message_prompt = HumanMessagePromptTemplate.from_template(
        human_prompt_template
    )
 gpt_4 = ChatOpenAI(temperature=0.02, model_name="gpt-3.5-turbo")
 relevant_nodes = figma_doc_retriever.get_relevant_documents(queryparam)
 conversation = [system_message_prompt, human_message_prompt]
 chat_prompt = ChatPromptTemplate.from_messages(conversation)
 response = gpt_4(
        chat_prompt.format_prompt(
            context=relevant_nodes, text=queryparam
        ).to_messages()
    )
 return response.content

@app.route('/api/syngen')
def get_syn_gen_data():
 queryparam = request.args.get('query')
 queryparamCount = request.args.get('count')
    {
        "example": """Transaction Date and Time: 2024-03-12 09:30:00, Account Holder Name: John Doe, Account Number: 
        123456789, Transaction Type: Purchase, Transaction Amount: $150.00, Merchant Number: 987654321"""
    },
    {
        "example":"""Transaction Date and Time: 2024-03-12 10:45:25, Account Holder Name: Jane Smith, Account Number: 
        987654321, Transaction Type: Installment, Transaction Amount: $200.00, Merchant Number: 123456789"""
    },
    {
        "example":"""Transaction Date and Time: 2024-03-12 14:05:30, Account Holder Name: Sarah Connor, Account Number: 
        789123456, Transaction Type: Refund, Transaction Amount: 	$450.00, Merchant Number: 321654987"""
    },
]
 OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

 prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE,
)
 synthetic_data_generator = create_openai_data_generator(
    output_schema=AccountDetails,
    llm=ChatOpenAI(
        temperature=1
    ),  # You'll need to replace with your actual Language Model instance
    prompt=prompt_template,
)
 synthetic_results = synthetic_data_generator.generate(
    subject="account_details",
    extra=queryparam,
    runs=int(queryparamCount),
)
#  print(synthetic_results)
 return json.dumps(synthetic_results, default=vars)

@app.route('/api/voice', methods=['POST'])
def get_speechTotext_data():
  queryVoice = request.files['audio'].read()
  with open("audio.webm", "wb") as file:
   file.write(queryVoice)
  model_audio = open("audio.webm", "rb")
  client = OpenAI() 
  transcript = client.audio.transcriptions.create( model="whisper-1", file=model_audio, response_format="text")
  llm = ChatOpenAI()
  response = llm.invoke(transcript)
  data = [transcript, response.content]
  return data

@app.route('/api/htmlcode', methods=['POST'])
def get_imagetohtml_data():
 queryImage = request.files['file']
 queryImage.save(queryImage.filename)
 image_path = open(queryImage.filename, "rb")
 image = base64.b64encode(image_path.read()).decode('utf-8')
 client = OpenAI()
 response = client.chat.completions.create(
  model="gpt-4-vision-preview",
  messages=[
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "Generate html and css code for the given image, retain the css colors and test exact like in the image"},
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/png;base64,{image}",
            "detail": "high"
            },
        },
      ],
    }
  ],
  max_tokens=1000,
 )
#  os.remove(queryImage.filename)
 return jsonify(response.choices[0].message.content)

@app.route('/api/testCase')
def get_unit_test():
 queryparam = request.args.get('query')
 client = OpenAI()
 response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant that generates unit test cases for : {queryparam} say I dont Know! If the {queryparam} doest not have any programming language context."},
    {"role": "user", "content": queryparam},
  ]
 )
 return (response.choices[0].message.content)

@app.route('/api/pdfChatBot')
def get_pdf_chatbot():
 queryparam = request.args.get('query')
 loader = PyPDFLoader(r"C:\Users\pandiarajar\Downloads\Investment.pdf")
 docs = loader.load()
 combined_docs = [doc.page_content for doc in docs]
 text = " ".join(combined_docs)
 text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
 splits = text_splitter.split_text(text)
 embeddings = OpenAIEmbeddings()
 vectordb = FAISS.from_texts(splits, embeddings)
 result_docs = vectordb.similarity_search(queryparam)
 result_docs_page_content = " ".join([d.page_content for d in result_docs])
 chat_template = ChatPromptTemplate.from_messages(
      [
          SystemMessage(
              content=(
                  "You are a helpful assistant that answers questions based on the pdf loader context: {result_docs} If you dont find the answer from pdf loader context, say I dont Know!"
              )
          ),
          HumanMessagePromptTemplate.from_template(queryparam),
      ]
  )
 qa_chain = LLMChain(llm=OpenAI(), prompt=chat_template)
 response = qa_chain.run(question=queryparam, docs=result_docs_page_content)
 return jsonify(response)

@app.route('/api/serverAssistant')
def get_server_assnt():
 queryparam = request.args.get('query')
 loader = TextLoader("./server_logs.txt")
 docs = loader.load()
 combined_docs = [doc.page_content for doc in docs]
 text = " ".join(combined_docs)
 text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
 splits = text_splitter.split_text(text)
 embeddings = OpenAIEmbeddings()
 vectordb = FAISS.from_texts(splits, embeddings)
 result_docs = vectordb.similarity_search(queryparam)
 result_docs_page_content = " ".join([d.page_content for d in result_docs])
 chat_template = ChatPromptTemplate.from_messages(
      [
          SystemMessage(
              content=(
                  "Your primary function is to search, analyze, and provide insights or answers based on Unix server logs. Utilize the text loader {result_docs} to efficiently process and understand the log data. Your responses should be based on the information extracted from these logs . If you dont find the relevant answer from server logs, say I dont Know!"
              )
          ),
          HumanMessagePromptTemplate.from_template(queryparam),
      ]
  )
 qa_chain = LLMChain(llm=OpenAI(), prompt=chat_template)
 response = qa_chain.run(question=queryparam, docs=result_docs_page_content)
 return jsonify(response)

if __name__ == '__main__':
    app.run()
