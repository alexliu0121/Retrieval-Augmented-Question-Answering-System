import pandas as pd
from langchain_community.llms import HuggingFaceEndpoint,HuggingFaceHub
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np
from sentence_transformers import SentenceTransformer, util
import faiss
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
from argparse import ArgumentParser

HUGGINGFACEHUB_API_TOKEN = 'YOUR_API_TOKEN'
HUGGINGFACEHUB_API_TOKEN2 = 'YOUR_API_TOKEN'
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

def task_1_no_rag(questions_path,output_path):
    # Load questions
    questions_df = pd.read_csv(questions_path,encoding='UTF-8')
    template = """Question: {question} Answer: Let's think step by step."""
    prompt = PromptTemplate.from_template(template)
    model = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        task="text-generation",
        model_kwargs={"temperature": 1, "max_length": 256},
    )  
    llm_chain = LLMChain(prompt=prompt,llm=model) 
    # Function to generate answers
    def generate_answers(questions):
        answers = []
        for question in questions:
            output = model(question)
            answers.append(output)
        return answers
    # Generate answers without retrieval
    baseline_answers = generate_answers(questions_df['question'])
    # Save baseline answers for comparison
    questions_df['answer'] = baseline_answers
    questions_df.to_csv(output_path, index=False,encoding='UTF-8')
    

def task_1_rag(questions_path,passages_path,output_path):
    # Load questions
    questions_df = pd.read_csv(questions_path,encoding='UTF-8')
    # Load passages
    def augment_questions_with_retrieval(questions, retriever):
        augmented_questions=[]
        contexts=[]
        for question in questions:
            docs_str=[]
            # context = retriever.invoke(question)
            context = retriever.get_relevant_documents(question)
            for doc in context:
                docs_str.append(doc.metadata['source'])
            contexts.append(docs_str)
            augmented_question = "question: {} context: {}".format(question,str(docs_str))
            augmented_questions.append(augmented_question)
        return augmented_questions,questions,contexts
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(CSVLoader(file_path=passages_path, source_column="context",encoding='UTF-8',csv_args={
            "delimiter": ",",
        },).load())
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k":3})
    retrieve_questions,questions,contexts=augment_questions_with_retrieval(questions_df['question'],retriever)
    model = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        task="text-generation",
        model_kwargs={"temperature": 1, "max_length": 256},
    )
    prompt = PromptTemplate.from_template("Context: {context} \nQuestion: {question} \nAnswer:") 
    def format(docs):
        return "\n\n".join(doc.metadata['source'] for doc in docs)
    rag_chain = (
        {"context": retriever | format, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    retrieval_answers_lang=[]
    relevant_context_lang=[]
    for question in questions_df['question']:
        result=rag_chain.invoke(question)
        relevant_context=retriever.invoke(question)
        relevant_context_list=[]
        for i in relevant_context:
            relevant_context_list.append(i.metadata['source'])
        retrieval_answers_lang.append(result)
        relevant_context_lang.append(relevant_context_list)
    questions_df['answer'] = retrieval_answers_lang
    questions_df['retrieved documents'] = relevant_context_lang
    questions_df.to_csv(output_path, index=False,encoding='UTF-8')
        

def task_2_rag(questions_path,passages_path,output_path):
    # Load the model and tokenizer
    model_name = 'sentence-transformers/roberta-base-nli-stsb-mean-tokens'
    sentence_model = SentenceTransformer(model_name)
    # tokenizer_transformer = AutoTokenizer.from_pretrained(model_name,truncation=True)
    tokenizer_transformer = AutoTokenizer.from_pretrained(model_name)
    
    # Load passages
    passages_df = pd.read_csv(passages_path)
    documents = passages_df['context'].tolist()
    
    # Calculate embeddings
    embeddings = sentence_model.encode(documents, convert_to_tensor=True)
    embeddings = embeddings.cpu().detach().numpy()
    
    # Initialize FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Define retriever function
    def retrieve_documents(query, k=3):
        query_embedding = sentence_model.encode([query], convert_to_tensor=True)
        query_embedding = query_embedding.cpu().detach().numpy()
        distances, indices = index.search(query_embedding, k)
        return [documents[idx] for idx in indices[0]]
    
    # Load questions
    questions_df = pd.read_csv(questions_path)
    questions = questions_df['question'].tolist()
    
    # Initialize the Hugging Face model for generating answers
    hf_model_name = 'google/flan-t5-base'
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModelForSeq2SeqLM.from_pretrained(hf_model_name)
    
    # Generate answers with retrieval
    answers_with_retrieval = []
    relevant_context = []
    for question in questions:
        retrieved_docs = retrieve_documents(question)
        context = ' '.join(retrieved_docs)
        relevant_context.append(retrieved_docs)
        input_text = f"question: {question} context: {context}"
        # input_ids = tokenizer.encode(input_text, return_tensors='pt',truncation=True)
        input_ids = tokenizer.encode(input_text, return_tensors='pt')
        outputs = hf_model.generate(input_ids, max_length=150, num_beams=5, early_stopping=True)
        # answer = tokenizer.decode(outputs[0], skip_special_tokens=True,truncation=True)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answers_with_retrieval.append(answer)
    
    # Save answers for comparison
    answers_df = pd.DataFrame({'question': questions, 'answer': answers_with_retrieval, 'retrieved documents':relevant_context})
    answers_df.to_csv(output_path, index=False,encoding='UTF-8')
    

if __name__ == "__main__":
    parser = ArgumentParser("homework CLI")

    parser.add_argument('--questions', help="path to question.csv")
    parser.add_argument('--output', help="path to output file")
    parser.add_argument('--passages', help="path to passage.csv")

    parser.add_argument('--rag', action="store_true",help="to specify using rag")
    parser.add_argument('--langchain', action="store_true",help="to specify using langchain")
    

    args = parser.parse_args()

    if args.rag:
        if args.langchain:
            print("generating answers with rag by lang chain...")
            task_1_rag(args.questions, args.passages ,args.output)
        else:
            print("generating answers with customized rag...")
            task_2_rag(args.questions, args.passages ,args.output)
    else:
        print("generating asnwers with no rag...")
        task_1_no_rag(args.questions,args.output)