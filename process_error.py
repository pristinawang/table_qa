import csv
import json
from datasets import load_dataset
import pandas as pd

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

# Load the TSV file
tsv_file = "./wikitq_pred/20241210114145.tsv"  # Replace with your TSV file path
json_file_path = "./wikitq_raw_out/20241210114145.json"  # Replace with your JSON file path
model_answer_file = "./wikitq_out/zeroshot/20241210114145.tsv"
dataset = load_dataset("Stanford/wikitablequestions")
dataset = dataset['train']

# Load model answers from the additional TSV file
model_answers = {}
with open(model_answer_file, 'r') as model_file:
    reader = csv.reader(model_file, delimiter='\t')
    for row in reader:
        model_answers[row[0]] = row[1]
# Load the JSON file
with open(json_file_path, 'r') as json_file:
    json_data = json.load(json_file)

# Process the TSV file
with open(tsv_file, 'r') as tsv:
    reader = csv.reader(tsv, delimiter='\t')
    for row in reader:
        data_id, status = row[0], row[1]

        # If the status is "False"
        if status == "False":
            # Find the example in the dataset with the matching ID
            example = next((item for item in dataset if item['id'] == data_id), None)

            # If a matching example is found
            if example:
                print(f"----------Id: {data_id}-----------------")
                print(">>Table:")
                print(TableToPIPE(T=example['table']))
                print("\n>>Question:")
                print(example['question'])
                print("\n>>Answer:")
                print(example['answers'])
                print("\n>>Model answer:")
                if data_id in model_answers:
                    print(model_answers[data_id])
                else:
                    print("No model answer found.")
                print("\n>>Output:")

                # Print outputs from the JSON file for the matching ID
                if data_id in json_data:
                    for i, out in enumerate(json_data[data_id]):
                        print(f"output {i + 1}:")
                        print(out)
                else:
                    print("No outputs found in JSON.")
