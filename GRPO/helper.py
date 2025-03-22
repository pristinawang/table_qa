import pandas as pd
from typing import List
import re
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

def process_answers(answers: List[str], target_delimiter:str = ", ") -> str:
    """
    Flatten the output for translation
    """
    output = target_delimiter.join(answers)
    if output.strip() == "":
        raise Exception("The Answer is EMPTY!")
    else:
        return output

def completions_to_answers(completions):
    # print('----Acc reward, Comple Format::---')
    # print(completions)
    # print('----------------')
    if isinstance(completions[0],str):
        completion_contents = completions  
    else:
        completion_contents = [completion[0]["content"] for completion in completions]
    
    # print('-----Acc reward, formatted completion-------')
    # print(completion_contents)
    # print('-------------------')
    matches = [re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL) for completion in completion_contents]
    contents = [match.group(1) if match else "" for match in matches]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return contents