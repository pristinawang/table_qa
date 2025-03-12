from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch
import pandas as pd
import re
from datetime import datetime
import os
import csv
import json

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

def get_prompts(dataset, error_dict, type):
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
    elif type=='error':
        data_dict={}
        for example in dataset:
            if example['id'] in error_dict:
                data_dict[example['id']]=example['question']
        system_prompt = {
            "role": "system",
            "content": "You are a helpful and concise assistant that answers questions accurately based on the provided context. You think step by step and give explanation and give only the final answer at last. Avoid any additional explanation or context in the final answer. You have to follow this format: 'Ans: <answer>'"
        }
        chat_prompts = [
            [
                system_prompt,  # System-level instruction
                {"role": "user", "content": "Given question and long answer below, respond with final answer according to the format: 'Ans: <answer>'"+'\nQuestion: '+data_dict[id]+'\nLong answer:'+error_dict[id]}
            ]
            for id in error_dict.keys()
        ]
        id_ls=[id for id in error_dict.keys()]
    return chat_prompts, id_ls

def write_err_ans(output_path, error_dict):
    for id in error_dict.keys():
        with open(output_path, 'a', newline='') as tsvfile:
            # Create a CSV writer object with tab as a delimiter
            writer = csv.writer(tsvfile, delimiter='\t')
            writer.writerow([id, 'Error'])
    
    print("Output all exported to path:", output_path)
        

def write_ans(output_path, error_path, results, id_ls, raw_out_dict):
    error_dict={}
    err_num=0
    #raw_out_dict={} #id:[out, out, ...,out]
    # Step 6: Process and display results
    for i, result in enumerate(results):
        id=id_ls[i]
        # print('Id:',id)
        # print(f"Prompt:\n{chat_prompts_small[i]}")
        # print(f"Generated Answer:\n{result.outputs[0].text}\n")
        if id not in raw_out_dict: raw_out_dict[id]=[result.outputs[0].text]
        else: raw_out_dict[id].append(result.outputs[0].text)
        match = re.search(r'Ans:\s*(.+)', result.outputs[0].text)
        if match:
            result = match.group(1).strip()  # Extract and strip spaces or newlines
            #print("Only Answer:\n"+result)
            with open(output_path, 'a', newline='') as tsvfile:
                # Create a CSV writer object with tab as a delimiter
                writer = csv.writer(tsvfile, delimiter='\t')
                writer.writerow([id, result])
        else:
            error_dict[id]=result.outputs[0].text
            print(">>Id: "+id+", Error No match found")
            err_num+=1
            error_content=">>Id: "+id+'\n'+result.outputs[0].text+'\n\n'
            ## output to error file
            with open(error_path, 'a') as file:  # Open the file in append mode
                file.write(error_content + "\n") 
            result='None'

    with open(error_path, 'a') as file:  # Open the file in append mode
        file.write("Error rate: "+ str(err_num/len(results))+'; Total data items: '+ str(len(results))+'; Total err #: '+ str(err_num))
    print("Output all exported to path:", output_path)
    print("Error logs to path:", error_path)
    return error_dict, raw_out_dict

if __name__=='__main__':
    print("-------------------")
    print('ZeroShot with temperature 0.9 and prompts twice for format issues.')
    print('--------------------')
    ## Check if we have cuda?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, "; using gpu:", torch.cuda.get_device_name())
    ## Output path
    now = datetime.now()
    # Format the datetime as numeric only: YYYYMMDDHHMMSS
    numeric_datetime = now.strftime("%Y%m%d%H%M%S")
    output_path = os.getcwd()+"/wikitq_out/zeroshot/" + numeric_datetime + ".tsv"
    error_path = os.getcwd()+"/wikitq_error_log/" + numeric_datetime + ".log"
    raw_out_path = os.getcwd()+"/wikitq_raw_out/" + numeric_datetime + ".json"
    print("Output Path:", output_path)
    print("Error Path:", error_path)
    print("Raw out path:", raw_out_path)
    ## dataset 
    dataset = load_dataset("Stanford/wikitablequestions")
    train_dataset = dataset['train']
    val_dataset = dataset['validation']
    test_dataset  = dataset['test']
    
    # Split dataset
    train_dataset=train_dataset.select(range(1,2500))
    print(train_dataset[0])
    print(train_dataset[4])
    print(train_dataset)
    chat_prompts, id_ls=get_prompts(dataset=train_dataset, error_dict=None, type='zeroshot')
    chat_prompts_small = chat_prompts[:5]
    id_ls_small = id_ls[:5]
    # Load model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

    llm = LLM(model=model_id, device=device)
    
    
    
    # Step 4: Define sampling parameters for inference
    # sampling_params = SamplingParams(
    #     temperature=0.7,
    #     top_p=0.9,
    #     max_tokens=100  # Limit the maximum token length of the response
    # )
    sampling_params=SamplingParams(temperature=0.9,max_tokens=1000, min_tokens=0)
    print('----------Chat prp small-----------')
    print(chat_prompts_small)
    print('--------------VVVV-----------------')

    # Step 5: Run inference with VLLM
    results = llm.chat(chat_prompts, sampling_params)
    
    # Write answers
    raw_out_dict={}
    error_dict, raw_out_dict=write_ans(output_path=output_path, error_path=error_path, results=results, id_ls=id_ls, raw_out_dict=raw_out_dict)
    
    # Reprompt for error items
    chat_prompts, id_ls=get_prompts(dataset=train_dataset, error_dict=error_dict, type='error')
    results = llm.chat(chat_prompts, sampling_params)
    error_path=os.getcwd()+"/wikitq_error_log/" + numeric_datetime +'_2'+ ".log"
    error_dict, raw_out_dict=write_ans(output_path=output_path, error_path=error_path, results=results, id_ls=id_ls, raw_out_dict=raw_out_dict)
    
    # Dump raw output to json path
    with open(raw_out_path, 'w') as json_file:
        json.dump(raw_out_dict, json_file, indent=4)

    print(f"Dictionary successfully dumped to {raw_out_path}")
    
    write_err_ans(output_path=output_path, error_dict=error_dict)