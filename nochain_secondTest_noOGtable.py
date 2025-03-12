from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch
import pandas as pd
import re
from datetime import datetime
import os
import csv

def TableToPIPE(T):
    '''
    parameter: og table format
    return table in PIPE format; data type is str
    '''
    # print('-------Table to Pipe')
    # print(T)
    df = pd.DataFrame(T['rows'], columns=T['header'])
    #print(df)
    dictTable = df.to_dict(orient="split")
    #print(dictTable)
    columns  =dictTable["columns"]
    data = dictTable["data"]
    strColumns = "col : "
    strData = ""

    
    for col in columns:
        strColumns = strColumns + col + " | "
    strColumns = strColumns[:-3]

    rowN = 1
    for row in data:
        strData = strData + "row" + str(rowN) + " : "
        for item in row:
            strData = strData + str(item) + " | "
        strData = strData[:-3]
        strData = strData + "\n"
        rowN += 1
    strData = strData[:-1]
    
    strPIPE = "/*\n" + strColumns + "\n" + strData + "\n*/"
    return strPIPE

def prompt_from_txt(file_path):


    # Open the file and read its content
    with open(file_path, 'r') as file:
        # Read the whole document as a string
        document_content = file.read()
    return document_content


def get_prompts(dataset, type, tableexp=None):
    if type=='zeroshot':
        
        # Step 3: Define the system prompt
        system_prompt = {
            "role": "system",
            "content": "You are a helpful and concise assistant that answers questions accurately based on the provided context. You think step by step and give explanation and give only the final answer at last. Avoid any additional explanation or context in the final answer. You have to follow this format: '<explanation> Ans: <answer>'"
        }

        # Step 4: Prepare chat prompts for each data point
        chat_prompts = [
            [
                system_prompt,  # System-level instruction
                {"role": "user", "content": 'Given the table below, answer the following question.\nTable:\n'+TableToPIPE(T=example['table'])+'\nQuestion: '+example['question']}
            ]
            for example in dataset
        ]
        id_ls=[example['id'] for example in dataset]
    elif type=='TableExpl':
        ## Get table description first
        system_prompt = {
            "role": "system",
            "content": "You are a helpful and concise assistant that answers questions accurately based on the provided context.'"
        }

        # Step 4: Prepare chat prompts for each data point
        userprompt=prompt_from_txt(file_path='./Prompts/summarize_table.txt')
        chat_prompts = [
            # Last time: 'Give detailed description and explain the table row by row and column by column. Explain the table for a reader who has access only to your explanation and not the table itself.\nTable:\n'+TableToPIPE(T=example['table'])
            [
                system_prompt,  # System-level instruction
                {"role": "user", "content": userprompt+'\nTable:\n'+TableToPIPE(T=example['table'])+'\nSummary:'}
            ]
            for example in dataset
        ]
        id_ls=[example['id'] for example in dataset]
    elif type=='zero_wTable' and tableexp is not None:
        system_prompt = {
            "role": "system",
            "content": "You are a helpful and concise assistant that answers questions accurately based on the provided context. You think step by step and give explanation and give only the final answer at last. Avoid any additional explanation or context in the final answer. You have to follow this format: '<explanation> Ans: <answer>'"
        }
        chat_prompts=[]
        for i, result in enumerate(tableexp):

            # print('Id:',id)
            # print(f"Prompt:\n{chat_prompts_small[i]}")
            # print(f"Generated Answer:\n{result.outputs[0].text}\n")
            description = result.outputs[0].text
            table=dataset[i]['table']
            question=dataset[i]['question']
            chat_prompts.append(
                [
                    system_prompt,  # System-level instruction
                    {"role": "user", "content": 'Given the table and table description below, answer the following question.\nTable:\n'+TableToPIPE(T=table)+'\nTable description:\n'+description+'\nQuestion: '+question}
                ]
            )
        id_ls=[example['id'] for example in dataset]
    elif type=='zero_wTable_NoOgTab' and tableexp is not None:
        system_prompt = {
            "role": "system",
            "content": "You are a helpful and concise assistant that answers questions accurately based on the provided context. You think step by step and give explanation and give only the final answer at last. Avoid any additional explanation or context in the final answer. You have to follow this format: '<explanation> Ans: <answer>'"
        }
        chat_prompts=[]
        for i, result in enumerate(tableexp):

            # print('Id:',id)
            # print(f"Prompt:\n{chat_prompts_small[i]}")
            # print(f"Generated Answer:\n{result.outputs[0].text}\n")
            summary = result.outputs[0].text
            table=dataset[i]['table']
            question=dataset[i]['question']
            chat_prompts.append(
                [
                    system_prompt,  # System-level instruction
                    {"role": "user", "content": 'Given summary below, answer the following question.\nSummary:\n'+summary+'\nQuestion: '+question}
                ]
            )
        id_ls=[example['id'] for example in dataset]
    return chat_prompts, id_ls

def write_ans(output_path, results, id_ls):
    #print('------------------------------------')
    # Step 6: Process and display results
    for i, result in enumerate(results):
        id=id_ls[i]
        # print('Id:',id)
        # print(f"Prompt:\n{chat_prompts_small[i]}")
        # print(f"Generated Answer:\n{result.outputs[0].text}\n")
        match = re.search(r'Ans:\s*(.+)', result.outputs[0].text)
        if match:
            result = match.group(1).strip()  # Extract and strip spaces or newlines
            #print("Only Answer:\n"+result)
        else:
            print(">>Id: "+id+", Error No match found")
            result='None'
        #print('------------------------------------')
        with open(output_path, 'a', newline='') as tsvfile:
            # Create a CSV writer object with tab as a delimiter
            writer = csv.writer(tsvfile, delimiter='\t')
            writer.writerow([id, result])
    print("Output all exported to path:", output_path)

if __name__=='__main__':
    ## Check if we have cuda?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, "; using gpu:", torch.cuda.get_device_name())
    ## Output path
    now = datetime.now()
    # Format the datetime as numeric only: YYYYMMDDHHMMSS
    numeric_datetime = now.strftime("%Y%m%d%H%M%S")
    output_path = os.getcwd()+"/wikitq_out/zeroshot/" + numeric_datetime + ".tsv"
    print("Output Path:", output_path)
    ## dataset 
    dataset = load_dataset("Stanford/wikitablequestions")
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset  = dataset['test']

    print(train_dataset[0])
    print(train_dataset[4])
    print(train_dataset)
    
    ## Split dataset
    train_dataset_small=train_dataset.select(range(1,2500))
    train_dataset_stand=train_dataset.select(range(1,7342))
    # Choose dataset to use
    train_dataset = train_dataset_small
    print('now dataset')
    print(train_dataset)
    # Load model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llm = LLM(model=model_id, device=device)
    sampling_params=SamplingParams(temperature=0.7, max_tokens=2048, min_tokens=0) #max_tokens=1000
    
    ## Experiment type
    type='zero_wTable_NoOgTab'
    if type=='zero_wTable_NoOgTab':
        #print('-------table prp-----------')
        chat_prompts, id_ls=get_prompts(dataset=train_dataset, type='TableExpl')
        print('-------table prp-----------')
        #print(chat_prompts)
        tableexp = llm.chat(chat_prompts, sampling_params)
    else: tableexp=None
    
    chat_prompts, id_ls=get_prompts(dataset=train_dataset, type=type, tableexp=tableexp)
    
    
    print('----------Chat prp-----------')
    #print(chat_prompts)
    #print('--------------VVVV-----------------')

    # Step 5: Run inference with VLLM
    results = llm.chat(chat_prompts, sampling_params)
    
    # Write answers
    write_ans(output_path=output_path, results=results, id_ls=id_ls)