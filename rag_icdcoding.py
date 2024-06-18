from dotenv import load_dotenv  
from langchain_community.document_loaders import DataFrameLoader 
from langchain_openai import OpenAIEmbeddings, ChatOpenAI  
from langchain_community.vectorstores.faiss import FAISS  
from langchain_core.prompts import ChatPromptTemplate  
from langchain_core.runnables import RunnablePassthrough  
from langchain_core.output_parsers import StrOutputParser  
from utils import parse_icd10_description  

def load_dfdata(df):
    """
    Load data from a DataFrame and return ICD descriptions.
    """
    loader = DataFrameLoader(df, page_content_column="ShortDescription")
    icd_descriptions = loader.load()
    return icd_descriptions

def create_vectorstore(doc):
    """
    Create a vector store from the provided documents using OpenAI embeddings.
    """
    embedding = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(doc, embedding=embedding)
    return vectorstore

def extract_assessment(llm, anote):
    """
    Extract the 'Assessment' section from a doctor's note using a language model.
    """
    # Define the prompt template to extract the Assessment section
    message = """
    Extract the 'Assessment' section from the following doctor's note:
    {note}
    Assessment:
    """
    prompt = ChatPromptTemplate.from_messages([("human", message)])

    # Create the chain to process the note
    chain = {"note": RunnablePassthrough()} | prompt | llm | StrOutputParser()

    return chain.invoke(anote)

def extract_demo(llm, vdb):
    """
    Extract demographic information (name and age) from a doctor's note.
    """
    message_demoinfo = """
    You are a professional medical coder. \
    From the provided note. \
    Extract the name of the patient. \
    Extract the age of the patient. \

    {note}
    """
    prompt_demo = ChatPromptTemplate.from_messages([("human", message_demoinfo)])

    # Create the chain to process the note for demographic information
    rag_chain_demo = {"note": RunnablePassthrough()} | prompt_demo | llm | StrOutputParser()

    return rag_chain_demo

def extract_icd(llm, vdb):
    """
    Infer the related ICD codes from the content of a doctor's note.
    """
    message_icd = """
    You are a professional medical coder. \
    From the provided note. \
    Infer the related ICD codes from the content

    {content}

    Context:
    {context}
    """
    prompt_icd = ChatPromptTemplate.from_messages([("human", message_icd)])

    # Create a retriever from the vector store
    retriever = vdb.as_retriever(search_kwargs={"k": 15})

    # Create the chain to process the note for ICD codes
    rag_chain_icd = {"context": retriever, "content": RunnablePassthrough()} | prompt_icd | llm | StrOutputParser()

    return rag_chain_icd

if __name__ == "__main__":
    load_dotenv()  # Load environment variables
    icdfile_path = './icd10cm_order_2024_top100.txt'  # Path to the ICD-10 descriptions file
    icd10_df = parse_icd10_description(icdfile_path)  # Parse the ICD-10 descriptions file
    data = load_dfdata(icd10_df)  # Load the data into a DataFrame
    db = create_vectorstore(data)  # Create a vector store from the data

    # Initialize the language model (LLM)
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0)

    # Create the chains for extracting demographic info and ICD codes
    chain_demo = extract_demo(llm, db)
    chain_icd = extract_icd(llm, db)

    # Example doctor's note for testing
    note1 = """
    Patient Name: John Doe1
    Date of Birth: 01/15/1980
    Visit Date: 06/11/2024
    Provider: Dr. Jane Smith

    Chief Complaint:
    Fever, abdominal pain, and diarrhea for five days.

    History of Present Illness:
    John Doe, 44, reports high fever, severe abdominal cramps, and persistent diarrhea. Symptoms began five days ago and have worsened. Denies recent travel but consumed undercooked poultry a week ago.

    Physical Examination:
        • Temperature: 102.2°F
        • Abdomen: Tenderness in the right lower quadrant with guarding.

    Assessment and Plan:
        1. typhoid fever.
        • Order blood cultures and stool tests.
        • Start empiric antibiotic therapy with ceftriaxone.
        2. Salmonella enteritis.
        • Maintain hydration with oral rehydration solutions.
        • Educate on safe food handling and proper cooking techniques.

    Follow-Up:
    Return in 48 hours for follow-up and test results review. Immediate return if symptoms worsen.

    Signature:
    Dr. Jane Smith, MD
    """
    # Extract the assessment from the note
    assessment = extract_assessment(llm, note1)
    print(chain_demo.invoke(note1))
    print(chain_icd.invoke(assessment))

    # Another example doctor's note for testing
    # note2 = """
    # Patient Name: John Doe2
    # Date of Visit: June 17, 2024

    # Chief Complaint: High fever, abdominal pain, and diarrhea.

    # History of Present Illness:
    # John Doe, 45, reports a high fever for 5 days, abdominal cramps, and watery diarrhea. Also complains of fatigue and body aches.

    # Review of Systems:
    #     • Fever, chills, fatigue
    #     • Abdominal pain, diarrhea
    #     • No cough, shortness of breath, chest pain

    # Physical Examination:
    #     • Temp: 39.5°C, BP: 120/80, HR: 110, RR: 18
    #     • Appears ill, febrile
    #     • Abdominal tenderness, hyperactive bowel sounds

    # Assessment:
    #     Suspected typhoid fever
    #     Possible acute amebic dysentery
    #     Calicivirus enteritis

    # Plan:
    #     • Tests: CBC, blood cultures, stool analysis, Widal test, PCR for viruses
    #     • Treatment: Rehydration, Ciprofloxacin, antipyretics, isolation
    #     • Follow-Up: Return in 48 hours for results review

    # Signature:
    # Dr. Jane Smith, MD
    # Internal Medicine
    # """ 

    # Extract the assessment from the note
    # assessment = extract_assessment(llm, note2)
    # print(chain_demo.invoke(note2))
    # print(chain_icd.invoke(assessment))