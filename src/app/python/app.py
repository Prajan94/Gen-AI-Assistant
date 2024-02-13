import fileinput
from flask import Flask, jsonify, request, current_app, make_response, abort
import json
from flask_cors import CORS, cross_origin
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from openai import OpenAI
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
class MedicalBilling(BaseModel):
    patient_id: int
    patient_name: str
    diagnosis_code: str
    procedure_code: str
    total_charge: float
    insurance_claim_amount: float
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
 print(response)
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
 figma_loader = FigmaFileLoader("figd_O43fiwMiHVK2oSRlyExiJQNA9YbpITjybico4VAO", "0-1", "QP4PgUhxDHl7IED9H4yc82")
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
 examples = [
    {
        "example": """Patient ID: 123456, Patient Name: John Doe, Diagnosis Code: 
        J20.9, Procedure Code: 99203, Total Charge: $500, Insurance Claim Amount: $350"""
    },
    {
        "example": """Patient ID: 789012, Patient Name: Johnson Smith, Diagnosis 
        Code: M54.5, Procedure Code: 99213, Total Charge: $150, Insurance Claim Amount: $120"""
    },
    {
        "example": """Patient ID: 345678, Patient Name: Emily Stone, Diagnosis Code: 
        E11.9, Procedure Code: 99214, Total Charge: $300, Insurance Claim Amount: $250"""
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
    output_schema=MedicalBilling,
    llm=ChatOpenAI(
        temperature=1
    ),  # You'll need to replace with your actual Language Model instance
    prompt=prompt_template,
)
 synthetic_results = synthetic_data_generator.generate(
    subject="medical_billing",
    extra=queryparam,
    runs=int(queryparamCount),
)
 return json.dumps(synthetic_results, default=vars)

@app.route('/api/voice', methods=['GET', 'POST'])
def get_speechTotext_data():
 if 'audio' in request.files:
            file = request.files['audio']
            # Get the file suffix based on the mime type.
            # print(file.mimetype)
            extname = file.mimetype
            if not extname:
                abort(400)

            # Test here for allowed file extensions.

            # Generate a unique file name with the help of consecutive numbering.
            i = 1
            while True:
                dst = os.path.join(
                    current_app.instance_path,
                    current_app.config.get('UPLOAD_FOLDER', 'uploads'),
                    secure_filename(f'audio_record_{i}{extname}'))
                if not os.path.exists(dst): break
                i += 1

            # Save the file to disk.
            file.save(dst)
            model_audio = open("dst", "rb")
            client = OpenAI()
            transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=model_audio,
            response_format="text",
            )
            print(transcript)
            return jsonify(transcript)
            # return make_response('', 200)
        
 abort(400)
 
if __name__ == '__main__':
    app.run()