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
      TransactionDateandTime: str
      AccountHolderName: str
      AccountNumber: int
      TransactionType: str
      TransactionAmount: float
      MerchantNumber: int
      # RecordID: int
      # PathType: str
      # PathID: str
      # PolicyNumber: str
      # PolicyRegionCode: str
      # ProductCode: str
      # ProductDescription: str
      # AccountName: str
      # PolicyCity: str
      # PolicyState: str
      # PolicyZipCode: str
      # InsuredYears: int
      # EarnedPremium: float
      # WrittenPremium: float
      # IncuredLosses: float
      # AccountLossRatio: float
      # ClaimStatus: str
      # ClaimNumber: str
      # CompanionClaimNumber: str
      # PolEffdate: str
      # PolExpDate: str
      # LossDate: str
      # ClaimReportedDate: str
      # ClaimClosedDate: str
      # CatastropheDescription: str
      # LossCauseDescription: str
      # LossTypeDescription: str
      # ClaimantName: str
      # AccidentState: str
      # PaidLoss: float
      # IndemnityPaidAmount: int
      # DCCPaidAmount: int
      # ANOPaidAmount: int
      # SalvageRecoveryAmount: int
      # SubrogationRecoveryAmount: float
      # PoliceAuthorityContacted: str
      # VehicleIdentificationNumber: str
      # VehicleLicenseNumber: str
      # PurposeofUse: str
      # DriverID: str
      # ClaimCoverageID: str
      # RiskUnitNumber: int
      # PerilCoverageCode: int
      # ClaimCoverage: int
      # LossRiskState: str
      # EnteredDate: str
      # UpdatedDate: str
      # EnteredBy: str
      # UpdatedBy: str
      # AdjusterID: str
      # AdjusterCd: str
      # LossAdjusterCompanyCd: str
      # LossLocation: str
      # VehicleModelYear: str
      # MakeofVehicle: str
      # ClaimantStory:str
      # LossCauseDescription: str
      # AdjusterNotes: str
      # RepairShopComments:  str
      # Fraud: str
      # FRAUDCATEGORY: str
      # STAGESINCLAIMLIFECYCLE: str
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
    #         {
    #   "example":
    #   """RecordID: 2015000921,
    #   PathType: RENEW,
    #   PathID: 7104761,
    #   PolicyNumber: PHPK914884,
    #   PolicyRegionCode: M,
    #   ProductCode: BA,
    #   ProductDescription: BUSINESS AUTO,
    #   AccountName: NORTH SALEM LIMOUSINE, INC,
    #   PolicyCity: NORTH SALEM,
    #   PolicyState: NY,
    #   PolicyZipCode: 10560,
    #   InsuredYears: 2,
    #   EarnedPremium: 11951,
    #   WrittenPremium: 11951,
    #   IncuredLosses: -6184.96,
    #   Account Loss Ratio: 294.8525,
    #   ClaimStatus: CLOSED,
    #   ClaimNumber: PHBA13060728640,
    #   CompanionClaim Number: NONE,
    #   Pol Eff date: ,
    #   Pol Exp Date: ,
    #   LossDate: 6/21/2013,
    #   ClaimReportedDate: 6/26/2013,
    #   ClaimClosedDate: 7/31/2013,
    #   CatastropheDescription: NONE,
    #   LossCauseDescription: HIT AND RUN BY OV,
    #   LossTypeDescription: COLLISION,
    #   ClaimantName: NORTH SALEM LIMOUSINE, INC.,
    #   AccidentState: NY,
    #   PaidLoss: -6184.96,
    #   IndemnityPaidAmount: 1000,
    #   DCCPaidAmount: 0,
    #   ANOPaidAmount: 0,
    #   SalvageRecovery Amount: 0,
    #   SubrogationRecovery Amount: -7184.96,
    #   Police Authority Contacted: NO,
    #   Vehicle Identification Number: KYVWY27A506030665,
    #   Vehicle License Number: 2ZBO276,
    #   Purpose of Use: Cargo Van,
    #   Driver ID: 2360313558,
    #   Claim Coverage ID: 190413233,
    #   Risk Unit Number: 119,
    #   Peril Coverage Code: 519,
    #   Claim Coverage: 2,
    #   Loss Risk State: AL,
    #   Entered Date: 9/17/2014,
    #   Updated Date: 6/9/2014,
    #   Entered By: Sidonia,
    #   Updated By: Jessie,
    #   Adjuster ID: 2002759157,
    #   Adjuster Cd: 52,
    #   Loss Adjuster Company Cd: 459,
    #   Loss Location: 947 Manley Center,
    #   Vehicle Model Year: 2005,
    #   Make of Vehicle: Chevrolet Express Cargo Van G2500,
    #   Claimant Story: A car suddenly jumped into my lane in front of the car ahead of me, causing it to apply brakes abruptly, which caused me to rear-end it. The car that jumped into the lane drove away, and the occupants of the rear-ended car all claim to have injuries.,
    #   Loss Cause Description: Rear-ended car in front of me,
    #   Adjuster Notes: The car responsible for causing this accident drove away.  No one called 911 ,
    #   Repair Shop Comments:  Vehicle left doors, left headlight & rear bumpers needs to be replaced and cannot be repaired. Total estimate for the expense is $23,560,
    #   Fraud: Yes,
    #   FRAUD CATEGORY: Hard fraud - Staged Crash,
    #   STAGES IN CLAIM LIFECYCLE: FNOL - Claimant Story. Adjudication - Adjuter Notes after physical visit; and checking Past history of the insured and vehicle for similar reported claims"""
    # },
    # {
    #   "example":
    #   """RecordID: 2015000921,
    #   PathType: RENEW,
    #   PathID: 7104761,
    #   PolicyNumber: PHPK914884,
    #   PolicyRegionCode: M,
    #   ProductCode: BA,
    #   ProductDescription: BUSINESS AUTO,
    #   AccountName: NORTH SALEM LIMOUSINE, INC,
    #   PolicyCity: NORTH SALEM,
    #   PolicyState: NY,
    #   PolicyZipCode: 10560,
    #   InsuredYears: 2,
    #   EarnedPremium: 11951,
    #   WrittenPremium: 11951,
    #   IncuredLosses: -6184.96,
    #   Account Loss Ratio: 294.8525,
    #   ClaimStatus: CLOSED,
    #   ClaimNumber: PHBA13060728640,
    #   CompanionClaim Number: NONE,
    #   Pol Eff date: ,
    #   Pol Exp Date: ,
    #   LossDate: 6/21/2013,
    #   ClaimReportedDate: 6/26/2013,
    #   ClaimClosedDate: 7/31/2013,
    #   CatastropheDescription: NONE,
    #   LossCauseDescription: HIT AND RUN BY OV,
    #   LossTypeDescription: COLLISION,
    #   ClaimantName: NORTH SALEM LIMOUSINE, INC.,
    #   AccidentState: NY,
    #   PaidLoss: -6184.96,
    #   IndemnityPaidAmount: 1000,
    #   DCCPaidAmount: 0,
    #   ANOPaidAmount: 0,
    #   SalvageRecovery Amount: 0,
    #   SubrogationRecovery Amount: -7184.96,
    #   Police Authority Contacted: NO,
    #   Vehicle Identification Number: KYVWY27A506030665,
    #   Vehicle License Number: 2ZBO276,
    #   Purpose of Use: Cargo Van,
    #   Driver ID: 2360313558,
    #   Claim Coverage ID: 190413233,
    #   Risk Unit Number: 119,
    #   Peril Coverage Code: 519,
    #   Claim Coverage: 2,
    #   Loss Risk State: AL,
    #   Entered Date: 9/17/2014,
    #   Updated Date: 6/9/2014,
    #   Entered By: Sidonia,
    #   Updated By: Jessie,
    #   Adjuster ID: 2002759157,
    #   Adjuster Cd: 52,
    #   Loss Adjuster Company Cd: 459,
    #   Loss Location: 947 Manley Center,
    #   Vehicle Model Year: 2005,
    #   Make of Vehicle: Chevrolet Express Cargo Van G2500,
    #   Claimant Story: I was driving at night on a 1 lane road, suddenly there was a improper turn on which another Vehicle hit me from the left and sped away. I don't have any police report for the incident. My left doors and headlight got damaged.,
    #   Loss Cause Description: HIT AND RUN BY OV,
    #   Adjuster Notes:  It was found that the insured has no police report & no eyewitness for the incident and also the accident was a hit and run case from another Vehicle which could not be traced.,
    #   Repair Shop Comments:  Vehicle left doors, left headlight & rear bumpers needs to be replaced and cannot be repaired. Total estimate for the expense is $23,560,
    #   Fraud : Yes,
    #   FRAUD CATEGORY: Hard fraud - Staged Crash,
    #   STAGES IN CLAIM LIFECYCLE: FNOL - Claimant Story. Adjudication - Adjuter Notes after physical visit, including work order details"""
    # },
    # {
    #   "example":
    #   """RecordID: 2015000921,
    #   PathType: NEW,
    #   PathID: 5230161,
    #   PolicyNumber: PHPK564006,
    #   PolicyRegionCode: M,
    #   ProductCode: BA,
    #   ProductDescription: BUSINESS AUTO,
    #   AccountName: CENTRAL LAUNDRY SERVICE CORP. & SEACREST LINEN,
    #   PolicyCity: BROOKLYN,
    #   PolicyState: NY,
    #   PolicyZipCode: 11225,
    #   InsuredYears: 1,
    #   EarnedPremium: 97019,
    #   WrittenPremium: 97019,
    #   IncuredLosses: 1198.75,
    #   Account Loss Ratio: 294.3782,
    #   ClaimStatus: CLOSED,
    #   ClaimNumber: PHBA11030535642,
    #   CompanionClaim Number: NONE,
    #   Pol Eff date: 1/1/2011,
    #   Pol Exp Date: 1/1/2012,
    #   LossDate: 6/21/2013,
    #   ClaimReportedDate: 6/26/2013,
    #   ClaimClosedDate: 3/30/2011,
    #   CatastropheDescription: NONE,
    #   LossCauseDescription: IV HIT WHILE PARKED,
    #   LossTypeDescription: COLLISION,
    #   ClaimantName: CENTRAL LAUNDRY SERVICE CORP. & SEA,
    #   AccidentState: NY,
    #   PaidLoss: 1198.75,
    #   IndemnityPaidAmount: 1198.75,
    #   DCCPaidAmount: 0,
    #   ANOPaidAmount: 106,
    #   SalvageRecovery Amount: 0,
    #   SubrogationRecovery Amount: 0,
    #   Police Authority Contacted: NO,
    #   Vehicle Identification Number: PFGVG15B946535762,
    #   Vehicle License Number: 4MQY668,
    #   Purpose of Use: Passenger Bus,
    #   Driver ID: 2360323485,
    #   Claim Coverage ID: 1262635885,
    #   Risk Unit Number: 767,
    #   Peril Coverage Code: 386,
    #   Claim Coverage: 2,
    #   Loss Risk State: AK,
    #   Entered Date: 9/17/2014,
    #   Updated Date: 7/20/2014,
    #   Entered By: Borg,
    #   Updated By: Giselle,
    #   Adjuster ID: 2002759153,
    #   Adjuster Cd: 80,
    #   Loss Adjuster Company Cd: 861,
    #   Loss Location: 8 Independence Center,
    #   Vehicle Model Year: 2008,
    #   Make of Vehicle: Ford E-540,
    #   Claimant Story: I was parking my Vehicle at night and due to dark I was not able to see the pillar. While backing up my Vehicle hit  the pillar badly damaging the rear side. ,
    #   Loss Cause Description: IV HIT WHILE PARKED,
    #   Adjuster Notes: Incident took place while parking  but there was no eyewitness for the same., Also when asked for appraisal vehicle was not readily available.,
    #   Repair Shop Comments:  Rear damaged badly and cost of replacing the spare parts of right side is $7500,
    #   Fraud : Yes,
    #   FRAUD CATEGORY: 1. Hard fraud - Staged Crash. DOL too close to policy inception date,
    #   STAGES IN CLAIM LIFECYCLE: FNOL - date of loss, pol eff date, and claimant Story\r\n2. Adjudication - Adjuter Notes after physical visit, including work order details"""
    # },
    # {
    #   "example":
    #   """RecordID: 2015000921,
    #   PathType: NEW,
    #   PathID: 5230161,
    #   PolicyNumber: PHPK564006,
    #   PolicyRegionCode: M,
    #   ProductCode: BA,
    #   ProductDescription: BUSINESS AUTO,
    #   AccountName: CENTRAL LAUNDRY SERVICE CORP. & SEACREST LINEN,
    #   PolicyCity: BROOKLYN,
    #   PolicyState: NY,
    #   PolicyZipCode: 11225,
    #   InsuredYears: 1,
    #   EarnedPremium: 97019,
    #   WrittenPremium: 97019,
    #   IncuredLosses: 500,
    #   Account Loss Ratio: 294.3782,
    #   ClaimStatus: CLOSED,
    #   ClaimNumber: PHBA11050547754,
    #   CompanionClaim Number: NONE,
    #   Pol Eff date: 1/1/2011,
    #   Pol Exp Date: 1/1/2012,
    #   LossDate: 4/26/2011,
    #   ClaimReportedDate: 5/5/2011,
    #   ClaimClosedDate: 5/17/2011,
    #   CatastropheDescription: NONE,
    #   LossCauseDescription: NONE,
    #   LossTypeDescription: COLLISION,
    #   ClaimantName: CENTRAL LAUNDRY SERVICE CORP. & SEA,
    #   AccidentState: NY,
    #   PaidLoss: 1992.41,
    #   IndemnityPaidAmount: 1992.41,
    #   DCCPaidAmount: 0,
    #   ANOPaidAmount: 374.95,
    #   SalvageRecovery Amount: 0,
    #   SubrogationRecovery Amount: 0,
    #   Police Authority Contacted: NO,
    #   Vehicle Identification Number: QJDNM22D664475374,
    #   Vehicle License Number: 1VEC133,
    #   Purpose of Use: Public Transportation,
    #   Driver ID: 2360316598,
    #   Claim Coverage ID: 884585790,
    #   Risk Unit Number: 530,
    #   Peril Coverage Code: 247,
    #   Claim Coverage: 1,
    #   Loss Risk State: AR,
    #   Entered Date: 4/8/2014,
    #   Updated Date: 4/16/2015,
    #   Entered By: Alvira,
    #   Updated By: Carlee,
    #   Adjuster ID: 2002759156,
    #   Adjuster Cd: 65,
    #   Loss Adjuster Company Cd: 939,
    #   Loss Location: 1 Meadow Valley Crossing,
    #   Vehicle Model Year: 2003,
    #   Make of Vehicle: Nova Bus Classic,
    #   Claimant Story: During dawn there was a heavy sandstorm. I was driving on a one lane road and I was not able to see another Vehicle approaching from other direction. We got a hit due to which my tires also got damaged.,
    #   Loss Cause Description: ,
    #   Adjuster Notes: No clear evidence of sandstorm from weather reports of the accident day. Two vehicles involved in accident. Insured Vehicle tires also got damaged. Claim has been filed very late after the loss date. The vehicle involved is a very old model.,
    #   Repair Shop Comments:  Bumper and engine got a hit. Needs to be replaced. Total Cost $45,000 approx.,
    #   Fraud : Yes,
    #   FRAUD CATEGORY: 1. Hard fraud - Staged Crash,
    #   STAGES IN CLAIM LIFECYCLE: FNOL - Claimant Story. Adjudication - Adjuter Notes after physical visit, including work order details"""
    # },
    # {
    #   "example":
    #   """RecordID: 2015000921,
    #   PathType: NEW,
    #   PathID: 5230161,
    #   PolicyNumber: PHPK564006,
    #   PolicyRegionCode: M,
    #   ProductCode: BA,
    #   ProductDescription: BUSINESS AUTO,
    #   AccountName: CENTRAL LAUNDRY SERVICE CORP. & SEACREST LINEN,
    #   PolicyCity: BROOKLYN,
    #   PolicyState: NY,
    #   PolicyZipCode: 11225,
    #   InsuredYears: 1,
    #   EarnedPremium: 97019,
    #   WrittenPremium: 97019,
    #   IncuredLosses: 500,
    #   Account Loss Ratio: 294.3782,
    #   ClaimStatus: CLOSED,
    #   ClaimNumber: PHBA11050547754,
    #   CompanionClaim Number: NONE,
    #   Pol Eff date: 1/1/2011,
    #   Pol Exp Date: 1/1/2012,
    #   LossDate: 4/26/2011,
    #   ClaimReportedDate: 5/5/2011,
    #   ClaimClosedDate: 10/18/2012,
    #   CatastropheDescription: NONE,
    #   LossCauseDescription: NONE,
    #   LossTypeDescription: COLLISION,
    #   ClaimantName: CENTRAL LAUNDRY SERVICE CORP. & SEA,
    #   AccidentState: NY,
    #   PaidLoss: 500,
    #   IndemnityPaidAmount: 500,
    #   DCCPaidAmount: 0,
    #   ANOPaidAmount: 0,
    #   SalvageRecovery Amount: 0,
    #   SubrogationRecovery Amount: 0,
    #   Police Authority Contacted: NO,
    #   Vehicle Identification Number: PHEOX90N945791752,
    #   Vehicle License Number: 8LDW344,
    #   Purpose of Use: Public Transportation,
    #   Driver ID: 2360309676,
    #   Claim Coverage ID: 1588607795,
    #   Risk Unit Number: 706,
    #   Peril Coverage Code: 883,
    #   Claim Coverage: 1,
    #   Loss Risk State: AR,
    #   Entered Date: 4/26/2015,
    #   Updated Date: 9/13/2014,
    #   Entered By: Brannon,
    #   Updated By: Nealon,
    #   Adjuster ID: 2002759156,
    #   Adjuster Cd: 75,
    #   Loss Adjuster Company Cd: 607,
    #   Loss Location: 1 Meadow Valley Crossing,
    #   Vehicle Model Year: 2003,
    #   Make of Vehicle: Nova Bus Classic,
    #   Claimant Story: A car jumped into my lane in front of the car ahead of me, causing it to apply brakes abruptly which caused me to rear-end it. Physical damage to both the cars. No injury to occupants.  ,
    #   Loss Cause Description: ,
    #   Adjuster Notes: The accident did take place as per police report and photographs of the accident scene.  Car has been taken for repairs to a workshop that is known for Counterfiet and inferior quality parts,
    #   Repair Shop Comments:  Bumper and engine got a hit. Needs to be replaced. Total Cost $25000 approx.,
    #   Fraud : Yes,
    #   FRAUD CATEGORY: Soft Fraud - Counterfeit parts,
    #   STAGES IN CLAIM LIFECYCLE: Adjudication - Adjuter Notes after physical visit, including workshop and repair order details"""
    # },
    # {
    #   "example":
    #   """RecordID: 2015000921,
    #   PathType: NEW,
    #   PathID: 5230161,
    #   PolicyNumber: PHPK564006,
    #   PolicyRegionCode: M,
    #   ProductCode: BA,
    #   ProductDescription: BUSINESS AUTO,
    #   AccountName: CENTRAL LAUNDRY SERVICE CORP. & SEACREST LINEN,
    #   PolicyCity: BROOKLYN,
    #   PolicyState: NY,
    #   PolicyZipCode: 11225,
    #   InsuredYears: 1,
    #   EarnedPremium: 97019,
    #   WrittenPremium: 97019,
    #   IncuredLosses: 0,
    #   Account Loss Ratio: 294.3782,
    #   ClaimStatus: CLOSED,
    #   ClaimNumber: PHBA11030538499,
    #   CompanionClaim Number: NONE,
    #   Pol Eff date: 1/1/2011,
    #   Pol Exp Date: 1/1/2012,
    #   LossDate: 3/18/2011,
    #   ClaimReportedDate: 3/21/2011,
    #   ClaimClosedDate: 3/30/2011,
    #   CatastropheDescription: NONE,
    #   LossCauseDescription: IV HIT PARKED OV,
    #   LossTypeDescription: COLLISION,
    #   ClaimantName: CENTRAL LAUNDRY SERVICE CORP. & SEA,
    #   AccidentState: NY,
    #   PaidLoss: 0,
    #   IndemnityPaidAmount: 0,
    #   DCCPaidAmount: 0,
    #   ANOPaidAmount: 0,
    #   SalvageRecovery Amount: 0,
    #   SubrogationRecovery Amount: 0,
    #   Police Authority Contacted: YES,
    #   Vehicle Identification Number: GTKUE77Y789299844,
    #   Vehicle License Number: 5WGQ690,
    #   Purpose of Use: Minivan,
    #   Driver ID: 2360320459,
    #   Claim Coverage ID: 690990033,
    #   Risk Unit Number: 507,
    #   Peril Coverage Code: 492,
    #   Claim Coverage: 1,
    #   Loss Risk State: CA,
    #   Entered Date: 5/1/2015,
    #   Updated Date: 2/13/2015,
    #   Entered By: Benjamin,
    #   Updated By: Fletch,
    #   Adjuster ID: 2002759154,
    #   Adjuster Cd: 47,
    #   Loss Adjuster Company Cd: 501,
    #   Loss Location: 79143 Lyons Park,
    #   Vehicle Model Year: 2007,
    #   Make of Vehicle: Ford Transit-25,
    #   Claimant Story: I was driving last night when my front left tire blowout due to which I was not able to control my Vehicle and hit a concrete structure on vehicle's left side. I fixed the tyre and drove to my destination, didn't call police.  Workshop repairs estimate is $39,000,
    #   Loss Cause Description: Collision with concrete structure caused by sudden burst of tyre,
    #   Adjuster Notes: Left side of the vehicle has dented quite a bit.  Since they fixed the tyre I don’t' have access to damaged/burst tyre.  Looking at dents closely, not all damages look fresh. ,
    #   Repair Shop Comments: Left side door replacement,
    #   Fraud : Yes,
    #   FRAUD CATEGORY: Hard fraud - Staged Crash,
    #   STAGES IN CLAIM LIFECYCLE:  Adjudication - Adjuter Notes after physical visit, including workshop and repair order details"""
    # },
    # {
    #   "example":
    #   """RecordID: 2015000921,
    #   PathType: RENEW,
    #   PathID: 5689050,
    #   PolicyNumber: PHPK711359,
    #   PolicyRegionCode: M,
    #   ProductCode: BA,
    #   ProductDescription: BUSINESS AUTO,
    #   AccountName: CENTRAL LAUNDRY SERVICE CORP. & SEACREST LINEN,
    #   PolicyCity: BROOKLYN,
    #   PolicyState: NY,
    #   PolicyZipCode: 11225,
    #   InsuredYears: 2,
    #   EarnedPremium: 102512,
    #   WrittenPremium: 102512,
    #   IncuredLosses: 0,
    #   Account Loss Ratio: 294.3782,
    #   ClaimStatus: CLOSED,
    #   ClaimNumber: PHBA11090584721,
    #   CompanionClaim Number: NONE,
    #   Pol Eff date: ,
    #   Pol Exp Date: ,
    #   LossDate: 9/20/2011,
    #   ClaimReportedDate: 9/23/2011,
    #   ClaimClosedDate: 10/10/2011,
    #   CatastropheDescription: NONE,
    #   LossCauseDescription: NONE,
    #   LossTypeDescription: COLLISION,
    #   ClaimantName: CENTRAL LAUNDRY SERVICE CORP &,
    #   AccidentState: NY,
    #   PaidLoss: 0,
    #   IndemnityPaidAmount: 0,
    #   DCCPaidAmount: 0,
    #   ANOPaidAmount: 0,
    #   SalvageRecovery Amount: 0,
    #   SubrogationRecovery Amount: 0,
    #   Police Authority Contacted: YES,
    #   Vehicle Identification Number: VYOTZ62J231602374,
    #   Vehicle License Number: 8YAI241,
    #   Purpose of Use: Passenger Bus,
    #   Driver ID: 2360304228,
    #   Claim Coverage ID: 1245452308,
    #   Risk Unit Number: 858,
    #   Peril Coverage Code: 903,
    #   Claim Coverage: 2,
    #   Loss Risk State: CO,
    #   Entered Date: 5/12/2014,
    #   Updated Date: 3/23/2015,
    #   Entered By: Maressa,
    #   Updated By: Raquel,
    #   Adjuster ID: 2002759158,
    #   Adjuster Cd: 42,
    #   Loss Adjuster Company Cd: 465,
    #   Loss Location: 73 Bluejay Court,
    #   Vehicle Model Year: 2009,
    #   Make of Vehicle: Ford E-540,
    #   Claimant Story: I left my vehicle at Bob's Car wash for washing and detailing.  They called me that there is a damage to windshield which is a tiny chip I had never noticed before.  They insisted and gave me good offer for windshield replacement.  I agreed.  Pls find attached bill and photographs,
    #   Loss Cause Description: ,
    #   Adjuster Notes: The windshield replacement done is not for a reputed vendor.  ,
    #   Repair Shop Comments: windshield replacement $6,000,
    #   Fraud : YES,
    #   FRAUD CATEGORY: Windshield Con,
    #   STAGES IN CLAIM LIFECYCLE: Adjudication - Adjuter Notes after physical visit, including workshop and repair order details"""
    # },
    # {
    #   "example":
    #   """RecordID: 2015000921,
    #   PathType: RENEW,
    #   PathID: 5689050,
    #   PolicyNumber: PHPK711359,
    #   PolicyRegionCode: M,
    #   ProductCode: BA,
    #   ProductDescription: BUSINESS AUTO,
    #   AccountName: CENTRAL LAUNDRY SERVICE CORP. & SEACREST LINEN,
    #   PolicyCity: BROOKLYN,
    #   PolicyState: NY,
    #   PolicyZipCode: 11225,
    #   InsuredYears: 2,
    #   EarnedPremium: 102512,
    #   WrittenPremium: 102512,
    #   IncuredLosses: 0,
    #   Account Loss Ratio: 294.3782,
    #   ClaimStatus: CLOSED,
    #   ClaimNumber: PHBA11120602494,
    #   CompanionClaim Number: NONE,
    #   Pol Eff date: ,
    #   Pol Exp Date: ,
    #   LossDate: 40889,
    #   ClaimReportedDate: 12/22/2011,
    #   ClaimClosedDate: 12/20/2011,
    #   CatastropheDescription: NONE,
    #   LossCauseDescription: LANE CHANGE - IMPROPER BY OV,
    #   LossTypeDescription: COLLISION,
    #   ClaimantName: CENTRAL LAUNDRY SERVICE CORP &,
    #   AccidentState: NY,
    #   PaidLoss: 0,
    #   IndemnityPaidAmount: 0,
    #   DCCPaidAmount: 0,
    #   ANOPaidAmount: 0,
    #   SalvageRecovery Amount: 0,
    #   SubrogationRecovery Amount: 0,
    #   Police Authority Contacted: NO,
    #   Vehicle Identification Number: NOCPD95E814467791,
    #   Vehicle License Number: 5FHG759,
    #   Purpose of Use: Passenger Bus,
    #   Driver ID: 2360375840,
    #   Claim Coverage ID: 566945214,
    #   Risk Unit Number: 406,
    #   Peril Coverage Code: 412,
    #   Claim Coverage: 2,
    #   Loss Risk State: CT,
    #   Entered Date: 9/8/2014,
    #   Updated Date: 9/25/2014,
    #   Entered By: Morgun,
    #   Updated By: Tucky,
    #   Adjuster ID: 2002759164,
    #   Adjuster Cd: 31,
    #   Loss Adjuster Company Cd: 273,
    #   Loss Location: 033 Linden Lane,
    #   Vehicle Model Year: 2005,
    #   Make of Vehicle: Ford E-540,
    #   Claimant Story: Yesterday I was returning from Passenger Bus to home during dusk, I was changing the lane to overtake the Vehicle ahead of me. Suddenly, MY right tire blowout and I lost control due to which I hit the divider. The car that was ahead of me did not stop.  I did not report to police.  Fixed the tire and drove to this workshop,
    #   Loss Cause Description: ,
    #   Adjuster Notes:  No eyewitness, no police report. Two Vehicles involved in the accident. Auto hit the divider. There was a delay in reporting of claim.,
    #   Repair Shop Comments:  Left side of the Vehicle is completely damaged. Total cost is $32000,
    #   Fraud : Yes,
    #   FRAUD CATEGORY: Hard fraud - Staged Crash,
    #   STAGES IN CLAIM LIFECYCLE: Adjudication - Adjuter Notes after physical visit, including workshop and repair order details"""
    # },
    # {
    #   "example":
    #   """RecordID: 2015000921,
    #   PathType: RENEW,
    #   PathID: 5689050,
    #   PolicyNumber: PHPK711359,
    #   PolicyRegionCode: M,
    #   ProductCode: BA,
    #   ProductDescription: BUSINESS AUTO,
    #   AccountName: CENTRAL LAUNDRY SERVICE CORP. & SEACREST LINEN,
    #   PolicyCity: BROOKLYN,
    #   PolicyState: NY,
    #   PolicyZipCode: 77586,
    #   InsuredYears: 2,
    #   EarnedPremium: 102512,
    #   WrittenPremium: 102512,
    #   IncuredLosses: 1116.48,
    #   Account Loss Ratio: 294.3782,
    #   ClaimStatus: CLOSED,
    #   ClaimNumber: PHBA11080573243,
    #   CompanionClaim Number: NONE,
    #   Pol Eff date: ,
    #   Pol Exp Date: ,
    #   LossDate: 8/8/2011,
    #   ClaimReportedDate: 8/11/2011,
    #   ClaimClosedDate: 9/20/2011,
    #   CatastropheDescription: NONE,
    #   LossCauseDescription: HIT IN SIDE BY OTHER VEHICLE,
    #   LossTypeDescription: COLLISION,
    #   ClaimantName: CENTRAL LAUNDRY SERVICE CORP &,
    #   AccidentState: NY,
    #   PaidLoss: 1116.48,
    #   IndemnityPaidAmount: 1116.48,
    #   DCCPaidAmount: 0,
    #   ANOPaidAmount: 813.55,
    #   SalvageRecovery Amount: 0,
    #   SubrogationRecovery Amount: 0,
    #   Police Authority Contacted: NO,
    #   Vehicle Identification Number: HWBOV15C722924268,
    #   Vehicle License Number: 3CXR500,
    #   Purpose of Use: Cargo Van,
    #   Driver ID: 2360276334,
    #   Claim Coverage ID: 2051108977,
    #   Risk Unit Number: 950,
    #   Peril Coverage Code: 606,
    #   Claim Coverage: 2,
    #   Loss Risk State: TX,
    #   Entered Date: 4/5/2015,
    #   Updated Date: 10/21/2014,
    #   Entered By: Jordan,
    #   Updated By: Jo-anne,
    #   Adjuster ID: 2002759150,
    #   Adjuster Cd: 51,
    #   Loss Adjuster Company Cd: 744,
    #   Loss Location: 62 Bellgrove Plaza,
    #   Vehicle Model Year: 2003,
    #   Make of Vehicle: Chevrolet Express Cargo Van G2500,
    #   Claimant Story:  I was close to my home last night when my tire blowout due to which I met with an accident and hit the other vehicle.  I didn't stop.  I have brought the vehicle to workshop,
    #   Loss Cause Description: HIT IN SIDE BY OTHER VEHICLE,
    #   Adjuster Notes: Two Vehicles involved in the accident. Other vehicle is not traceable. Accident took place near insured residence. No police report & eyewitness found.  The workshop being used for repairs is not trusworthy.  I see the work order having unrelated repair work as wel,
    #   Repair Shop Comments: ,
    #   Fraud : Yes,
    #   FRAUD CATEGORY: Soft Fraud - Shop not trustworthy,
    #   STAGES IN CLAIM LIFECYCLE: Adjudication - Adjuter Notes after physical visit, including workshop and repair order details"""
    # },
    # {
    #   "example":
    #   """RecordID: 2015000921,
    #   PathType: NEW,
    #   PathID: 6545313,
    #   PolicyNumber: PHPK811863,
    #   PolicyRegionCode: NE,
    #   ProductCode: BA,
    #   ProductDescription: BUSINESS AUTO,
    #   AccountName: TRI-STATE OF BRANFORD LLC,
    #   PolicyCity: BRANFORD,
    #   PolicyState: CT,
    #   PolicyZipCode: 6405,
    #   InsuredYears: 1,
    #   EarnedPremium: 23658,
    #   WrittenPremium: 23658,
    #   IncuredLosses: 570.17,
    #   Account Loss Ratio: 246.6188,
    #   ClaimStatus: CLOSED,
    #   ClaimNumber: PHBA12030622672,
    #   CompanionClaim Number: NONE,
    #   Pol Eff date: ,
    #   Pol Exp Date: ,
    #   LossDate: 3/20/2012,
    #   ClaimReportedDate: 3/22/2012,
    #   ClaimClosedDate: 4/24/2012,
    #   CatastropheDescription: NONE,
    #   LossCauseDescription: NONE,
    #   LossTypeDescription: COLLISION,
    #   ClaimantName: TRI-STATE OF BRANFORD, LLC.,
    #   AccidentState: CT,
    #   PaidLoss: 570.17,
    #   IndemnityPaidAmount: 570.17,
    #   DCCPaidAmount: 0,
    #   ANOPaidAmount: 0,
    #   SalvageRecovery Amount: 0,
    #   SubrogationRecovery Amount: 0,
    #   Police Authority Contacted: NO,
    #   Vehicle Identification Number: ZLIXK46D625042744,
    #   Vehicle License Number: 7XTU885,
    #   Purpose of Use: Cargo Van,
    #   Driver ID: 2360331918,
    #   Claim Coverage ID: 359547925,
    #   Risk Unit Number: 288,
    #   Peril Coverage Code: 91,
    #   Claim Coverage: 1,
    #   Loss Risk State: FL,
    #   Entered Date: 10/28/2014,
    #   Updated Date: 8/23/2014,
    #   Entered By: Pearl,
    #   Updated By: Georgina,
    #   Adjuster ID: 2002759148,
    #   Adjuster Cd: 68,
    #   Loss Adjuster Company Cd: 709,
    #   Loss Location: 9 Arrowood Parkway,
    #   Vehicle Model Year: 2000,
    #   Make of Vehicle: Chevrolet Express Cargo Van G2500,
    #   Claimant Story: It was broad daylight I was tailgating the Vehicle ahead of me near my residence. I was not able to see any intersection due to which I hit lamppost. I got some injuries on my knee and head. There was no eyewitness and police report of this incidence. ,
    #   Loss Cause Description: ,
    #   Adjuster Notes: Story told by claimant seems to have gaps. Looks like Lies about location of the accident,
    #   Repair Shop Comments:  Vehicle windshield and bumper needs to be replaced. Total Expense is $15760,
    #   Fraud : Yes,
    #   FRAUD CATEGORY: Soft fraud - Lie about location,
    #   STAGES IN CLAIM LIFECYCLE: Adjudication - Adjuter Notes after physical visit, including workshop and repair order details"""
    # },
    # {
    #   "example":
    #   """RecordID: 2015000921,
    #   PathType: NEW,
    #   PathID: 6545313,
    #   PolicyNumber: PHPK811863,
    #   PolicyRegionCode: NE,
    #   ProductCode: BA,
    #   ProductDescription: BUSINESS AUTO,
    #   AccountName: TRI-STATE OF BRANFORD LLC,
    #   PolicyCity: BRANFORD,
    #   PolicyState: CT,
    #   PolicyZipCode: 6405,
    #   InsuredYears: 1,
    #   EarnedPremium: 23658,
    #   WrittenPremium: 23658,
    #   IncuredLosses: 328.54,
    #   Account Loss Ratio: 246.6188,
    #   ClaimStatus: CLOSED,
    #   ClaimNumber: PHBA12090664860,
    #   CompanionClaim Number: NONE,
    #   Pol Eff date: ,
    #   Pol Exp Date: ,
    #   LossDate: 9/19/2012,
    #   ClaimReportedDate: 9/20/2012,
    #   ClaimClosedDate: 9/27/2012,
    #   CatastropheDescription: NONE,
    #   LossCauseDescription: PARKED IV - IV STRUCK WHILE PARKED,
    #   LossTypeDescription: COLLISION,
    #   ClaimantName: TRI-STATE OF BRANFORD, LLC.,
    #   AccidentState: CT,
    #   PaidLoss: 328.54,
    #   IndemnityPaidAmount: 328.54,
    #   DCCPaidAmount: 0,
    #   ANOPaidAmount: 98,
    #   SalvageRecovery Amount: 0,
    #   SubrogationRecovery Amount: 0,
    #   Police Authority Contacted: NO,
    #   Vehicle Identification Number: VEISG27Z517426728,
    #   Vehicle License Number: 5WRV485,
    #   Purpose of Use: Cargo Van,
    #   Driver ID: 2360280554,
    #   Claim Coverage ID: 2147867573,
    #   Risk Unit Number: 715,
    #   Peril Coverage Code: 657,
    #   Claim Coverage: 1,
    #   Loss Risk State: GA,
    #   Entered Date: 11/23/2014,
    #   Updated Date: 3/25/2015,
    #   Entered By: Gloriana,
    #   Updated By: Dinnie,
    #   Adjuster ID: 2002759164,
    #   Adjuster Cd: 41,
    #   Loss Adjuster Company Cd: 522,
    #   Loss Location: 3 Southridge Street,
    #   Vehicle Model Year: 2004,
    #   Make of Vehicle: Chevrolet Express Cargo Van G2500,
    #   Claimant Story: Yesterday I was returning from site to home during dusk. While taking a turn I collided with another Vehicle's boot. My Vehicle got some damages and was not drive-able.\r\nI had to call the towing service to take my Vehicle to the repair station. There were no eye witness. ,
    #   Loss Cause Description: PARKED IV - IV STRUCK WHILE PARKED,
    #   Adjuster Notes: The other vehicle looks like a ghost vehicle.   No eyewitness & police report found. Insured have injuries on knee & head. Property damagedon't match with story how it happned,
    #   Repair Shop Comments: ,
    #   Fraud : Yes,
    #   FRAUD CATEGORY: Hard fraud - Staged Crash,
    #   STAGES IN CLAIM LIFECYCLE: Adjudication - Adjuter Notes after physical visit, including workshop and repair order details"""
    # }
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