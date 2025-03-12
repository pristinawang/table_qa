from datetime import datetime
import csv
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import transformers
import torch
import re
import os
import copy

def f_add_column(df,columnHeader,columnData):
    '''
    return: dataframe
    '''
    # Add the list as a new column
    try:
        df[columnHeader] = columnData
        return df
    except Exception as e:
        return None

def f_select_column(df,colsToBeSelected):
    '''
    parameter: colsToBeSelected is a list with column names that need to be selected
    '''
    try:
        return df[colsToBeSelected]
    except Exception as e:
        return None

def f_select_row(df,rows_to_be_selected):
    '''
    parameter: rowsToBeSelected is a list with row names that need to be selected
                expect items in list to be str, i.e. "row1", "row2", etc
                
                switching to getting integer list
    '''
    try:
        rowsToBeSelected = rows_to_be_selected.copy()
        for i in range(len(rowsToBeSelected)):
            # row = rowsToBeSelected[i]
            # ## Error prone
            # row = int(row[3:]) - 1
            # rowsToBeSelected[i] = row
            row = rowsToBeSelected[i]
            rowsToBeSelected[i] = row-1
        return df.iloc[rowsToBeSelected]  
    except Exception as e:
        return None

def f_group_by(df, column):
    '''
    From the paper:
    Groups the rows by the contents of a specific column and provides the count
    of each enumeration value in that column. Many table-based questions or statements involve
    counting, but LLMs are not proficient at this task.
    ----------------------------------------
    parameters: column is one of the column value in the df 
    ''' 
    try:
        return df.groupby(column)[column].count().reset_index(name='Count')
    except Exception as e:
        return None

def f_sort_by(df, column, order):
    '''
    Description(from the paper):
    sorts the rows based on the contents of a specific column. When dealing with
    questions or statements involving comparison or extremes, LLMs can utilize this operation to
    rearrange the rows. The relationship can be readily inferred from the order of the sorted rows
    ----------------------------------------
    parameters: 
    column: the column that we are sorting by
    order: order of the sorting
    '''
    try:
        if order == "Ascend":
            ascending = True
        else:
            ascending = False
        return df.sort_values(by=column, ascending=ascending)
    except Exception as e:
        return None

def dfToPIPE(df):
    '''
    return table in PIPE format; data type is str
    '''
    dictTable = df.to_dict(orient="split")
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

def dataset():
    dataset = load_dataset("Stanford/wikitablequestions")
    trainDataset = dataset['train']
    valDataset = dataset['validation']
    testDataset  = dataset['test']
    return trainDataset, valDataset, testDataset
    

class ChainOfTable:
    def __init__(self, train_dataset, llm, model, tokenizer, device, output_path, anal_path, debug=False, debug_acc=False):
        self.output_path = output_path
        self.anal_path = anal_path
        self.llm=llm
        self.debug=debug
        self.debug_acc=debug_acc
        self.train_dataset = train_dataset
        self.model=model
        self.tokenizer=tokenizer
        self.device=device
        self.prompt_f_add_column = 'To answer the question, we can first use f_add_column() to add more columns to the table.\n'+\
                'The added columns should have these data types:\n'+\
                '1. Numerical: the numerical strings that can be used in sort, sum\n'+\
                '2. Datetype: the strings that describe a date, such as year, month, day\n'+\
                '3. String: other strings\n'+\
                '/*\n'+\
                'col : Week | When | Kickoff | Opponent | Results; Final score | Results; Team record\n'+\
                'row 1 : 1 | Saturday, April 13 | 7:00 p.m. | at Rhein Fire | W 27-21 | 1-0\n'+\
                'row 2 : 2 | Saturday, April 20 | 7:00 p.m. | London Monarchs | W 37-3 | 2-0\n'+\
                'row 3 : 3 | Sunday, April 28 | 6:00 p.m. | at Barcelona Dragons | W 33-29 | 3-0\n'+\
                '*/\n'+\
                'Question: what is the date of the competition with highest attendance?\n'+\
                'The existing columns are: "Week", "When", "Kickoff", "Opponent", "Results; Final score",\n'+\
                '"Results; Team record", "Game site", "Attendance".\n'+\
                'Explanation: the question asks about the date of the competition with highest score. Each\n'+\
                'row is about one competition. We extract the value from column "Attendance" and create a\n'+\
                'different column "Attendance number" for each row. The datatype is Numerical.\n'+\
                'Therefore, the answer is: f_add_column(Attendance number). The value: 32092 | 34186 | 17503\n'+\
                '/*\n'+\
                'col : Rank | Lane | Player | Time\n'+\
                'row 1 : | 5 | Olga Tereshkova (KAZ) | 51.86\n'+\
                'row 2 : | 6 | Manjeet Kaur (IND) | 52.17\n'+\
                'row 3 : | 3 | Asami Tanno (JPN) | 53.04\n'+\
                '*/\n'+\
                'Question: tell me the number of athletes from japan.\n'+\
                'The existing columns are: Rank, Lane, Player, Time.\n'+\
                'Explanation: the question asks about the number of athletes from japan. Each row is about\n'+\
                'one athlete. We need to know the country of each athlete. We extract the value from column\n'+\
                '"Player" and create a different column "Country of athletes" for each row. The datatype\n'+\
                'is String.\n'+\
                'Therefore, the answer is: f_add_column(Country of athletes). The value: KAZ | IND | JPN'
        self.prompt_f_select_column = 'Use f_select_column() to filter out useless columns in the table according to information\n'+\
                'in the question and the table.\n'+\
                '/*\n'+\
                'table_caption : south wales derby\n'+\
                'col : competition | total matches | cardiff win | draw | swansea win\n'+\
                'row 1 : league | 55 | 19 | 16 | 20\n'+\
                'row 2 : fa cup matches | 2 | 0 | 27 | 2\n'+\
                'row 3 : league cup | 5 | 2 | 0 | 3\n'+\
                '*/\n'+\
                'Question: what is the competition with the highest number of total matches?\n'+\
                'Explanation: the question wants to know what competition has the highest number of total matches.\n'+\
                'We need to know each competition\'s total matches so we extract "competition" and "total matches".\n'+\
                'The answer is : f_select_column([competition, total matches])'
        self.prompt_f_select_row = 'Using f_select_row() to select relevant rows in the given table that answers the given\n'+\
                'question.\n'+\
                'Please use f_select_row([*]) to select all rows in the table.\n'+\
                '/*\n'+\
                'table caption : 1972 vfl season.\n'+\
                'col : home team | home team score | away team | away team score | venue | crowd\n'+\
                'row 1 : st kilda | 13.12 (90) | melbourne | 13.11 (89) | moorabbin oval | 18836\n'+\
                'row 2 : south melbourne | 9.12 (66) | footscray | 11.13 (79) | lake oval | 9154\n'+\
                'row 3 : richmond | 20.17 (137) | fitzroy | 13.22 (100) | mcg | 27651\n'+\
                'row 4 : geelong | 17.10 (112) | collingwood | 17.9 (111) | kardinia park | 23108\n'+\
                'row 5 : north melbourne | 8.12 (60) | carlton | 23.11 (149) | arden street oval | 11271\n'+\
                'row 6 : hawthorn | 15.16 (106) | essendon | 12.15 (87) | vfl park | 36749\n'+\
                '*/\n'+\
                'Question : what is the away team with the highest score?\n'+\
                'explain : the statement want to ask the away team of highest away team score. the highest\n'+\
                'away team score is 23.11 (149). it is on the row 5.so we need row 5.\n'+\
                'The answer is : f_select_row([row 5])'
        self.prompt_f_group_by='To answer the question, we can first use f_group_by() to group the values in a column.\n'+\
                '/*\n'+\
                'col : Rank | Lane | Athlete | Time | Country\n'+\
                'row 1 : 1 | 6 | Manjeet Kaur (IND) | 52.17 | IND\n'+\
                'row 2 : 2 | 5 | Olga Tereshkova (KAZ) | 51.86 | KAZ\n'+\
                'row 3 : 3 | 4 | Pinki Pramanik (IND) | 53.06 | IND\n'+\
                'row 4 : 4 | 1 | Tang Xiaoyin (CHN) | 53.66 | CHN\n'+\
                'row 5 : 5 | 8 | Marina Maslyonko (KAZ) | 53.99 | KAZ\n'+\
                '*/\n'+\
                'Question: tell me the number of athletes from japan.\n'+\
                'The existing columns are: Rank, Lane, Athlete, Time, Country.\n'+\
                'Explanation: The question asks about the number of athletes from India. Each row is about\n'+\
                'an athlete. We can group column "Country" to group the athletes from the same country.\n'+\
                'Therefore, the answer is: f_group_by(Country).'
        self.prompt_f_sort_by='To answer the question, we can first use f_sort_by() to sort the values in a column to get\n'+\
                'the order of the items. The order can be "large to small" or "small to large".\n'+\
                'The column to sort should have these data types:\n'+\
                '1. Numerical: the numerical strings that can be used in sort\n'+\
                '2. DateType: the strings that describe a date, such as year, month, day\n'+\
                '3. String: other strings\n'+\
                '/*\n'+\
                'col : Position | Club | Played | Points | Wins | Draws | Losses | Goals for | Goals against\n'+\
                'row 1 : 1 | Malaga CF | 42 | 79 | 22 | 13 | 7 | 72 | 47\n'+\
                'row 10 : 10 | CP Merida | 42 | 59 | 15 | 14 | 13 | 48 | 41\n'+\
                'row 3 : 3 | CD Numancia | 42 | 73 | 21 | 10 | 11 | 68 | 40\n'+\
                '*/\n'+\
                'Question: what club placed in the last position?\n'+\
                'The existing columns are: Position, Club, Played, Points, Wins, Draws, Losses, Goals for,\n'+\
                'Goals against\n'+\
                'Explanation: the question asks about the club in the last position. Each row is about a\n'+\
                'club. We need to know the order of position from last to front. There is a column for\n'+\
                'position and the column name is Position. The datatype is Numerical.\n'+\
                'Therefore, the answer is: f_sort_by(Position), the order is "large to small".'
    def chain_of_table(self,T,Q):
        chain = []
        chain_dict = {}
        operations = set()
        # Get dataframe
        df = pd.DataFrame(T['rows'], columns=T['header'])
        df = df.map(lambda x: pd.to_numeric(x, errors='ignore') if isinstance(x, str) else x)
        print("LET's check df")
        print(T)
        print(df)
        
        print(list(df.select_dtypes(include=['number']).columns))
        print('~~~~~~~~~~~~~~~~~~~~~')
        round=0
        while True:
            if self.debug:
                print("Round", round, "Chain", chain)
            f = self.DynamicPlan(T=df, Q=Q, chain=chain, operations=operations)
            if f=="": 
                if self.debug: print("f is empty")
                break
            elif f=="<END>": 
                if self.debug:
                    print("#########################")
                    print('LAST end f',f)
                chain.append(f)
                operations.add(f)
                chain_dict[f]=f
                break
            
            if self.debug:
                print("#########################")
                print('new f',f)
                print("#########################")
            args = self.GenerateArgs(T=df, Q=Q, f=f)
            if args==None: 
                if self.debug: print("args is None")
                break
            if self.debug:
                print("#########################")
                print("OG ARGSS")
                print(args)
            # Update table
            if self.debug:
                print("#########################")
                print("OLD DF")
                print(df)
            old_df = df
            df = self.f(df=df, f=f, args=args)
            if not(isinstance(df, pd.DataFrame)):
                if self.debug: print("df is gibberish")
                df = old_df
                break
            if self.debug:
                print("NEW DF")
                print(df)
                print("#########################")
                print("check type")
                print(args)
            if isinstance(args[0], list): 
                if not(isinstance(args[0][0],str)):
                    args[0]=[str(x) for x in args[0]]
                argument = ', '.join(args[0])
            else: argument = args[0]
            operations.add(f)
            chain_dict[f]=f+"("+argument+")"
            f=f+"("+argument+")"
            if self.debug:
                print("#########Old chain################")
                print(chain)
            chain.append(f)
            if len(chain)>=6: 
                print("Used up all operations")
                print("PRINt chain", chain)
                break
            if self.debug:
                print("#########Append chain################")
                print(chain)
            round+=1
            
        
        raw_a=self.Query(T=df, Q=Q, chain=chain, original_table=T)
        pattern = "^(.*)"
        a = re.search(pattern, raw_a) ##.group(1)
        if a==None: return "",chain_dict,operations
        else: 
            a = a.group(1)
            a = a.strip(" ")
            return a,chain_dict, operations

    def Query(self, T, Q, chain, original_table):
        '''
        T takes pandas dataframe and is the newly constructed dataframe
        original table is the table that comes with the dataset
        Q: question or statement string
        '''
        # Get dataframe
        df = pd.DataFrame(original_table['rows'], columns=original_table['header'])
        # Get PIPE table form
        new_table_pipe = dfToPIPE(df=T)
        old_table_pipe = dfToPIPE(df=df)
        examples='Here is the table to answer this question. Please understand the table and answer the\n'+\
                'question:\n'+\
                '/*\n'+\
                'col : Rank | City | Passengers Number | Ranking | Airline\n'+\
                'row 2 : 2 | United States, Houston | 5465 | 8 | United Express\n'+\
                'row 3 : 3 | Canada, Calgary | 3761 | 5 | Air Transat, WestJet\n'+\
                'row 4 : 4 | Canada, Saskatoon | 2282 | 4 |\n'+\
                'row 5 : 5 | Canada, Vancouver | 2103 | 2 | Air Transat\n'+\
                'row 6 : 6 | United States, Phoenix | 1829 | 1 | US Airways\n'+\
                'row 7 : 7 | Canada, Toronto | 1202 | 1 | Air Transat, CanJet\n'+\
                'row 8 : 8 | Canada, Edmonton | 110 | 2 |\n'+\
                'row 9 : 9 | United States, Oakland | 107 | 5 |\n'+\
                '*/\n'+\
                'Question: how many more passengers flew to los angeles than to saskatoon from manzanillo\n'+\
                'airport in 2013?\n'+\
                'The anwser is: 12467\n'+\
                'Here is the table to answer this question. Please understand the table and answer the\n'+\
                'question:\n'+\
                '/*\n'+\
                'col : Rank | Country\n'+\
                'row 1 : 1 | ESP\n'+\
                'row 2 : 2 | RUS\n'+\
                'row 3 : 3 | ITA\n'+\
                'row 4 : 4 | ITA\n'+\
                'row 5 : 5 | ITA\n'+\
                'row 6 : 6 | RUS\n'+\
                'row 7 : 7 | ESP\n'+\
                'row 8 : 8 | FRA\n'+\
                'row 9 : 9 | ESP\n'+\
                'row 10 : 10 | FRA\n'+\
                '*/\n'+\
                'Performed operations and got the following table.\n'+\
                'Operations performed: f_group_by(Country)\n'+\
                '/*\n'+\
                'Group ID | Country | Count\n'+\
                '1 | ITA | 3\n'+\
                '2 | ESP | 3\n'+\
                '3 | RUS | 2\n'+\
                '4 | FRA | 2\n'+\
                '*/\n'+\
                'Question: which country had the most cyclists in top 10?\n'+\
                'The answer is: Italy and Spain'
        question = "Question: " + Q
        instruction = 'Here is the table to answer this question. Please understand the table and answer the\n'+\
                'question:'
        instruction2 = 'Performed operations and got the following table.'
        operations = 'Operations performed: ' + self.chain_to_string(chain=chain)[:-4] #[:-4] to strip the last " -> " 
        answer_prompt = "The answer is: "
        prompt = examples + "\n" + instruction + "\n" + old_table_pipe + "\n" + instruction2 + "\n" + operations + "\n" + new_table_pipe + "\n" + question + "\n" + answer_prompt
        #raw_answer = self.prompt_model(prompt=prompt, model=self.model, tokenizer=self.tokenizer, device=self.device)
        raw_answer = self.prompt_vllm(prompt=prompt, llm=self.llm, type="ans")
        if self.debug:
            print("#########################")
            print("RawAns")
            print(raw_answer)
        ans = raw_answer.strip(' ').strip('\n')
        if self.debug:
            print("CleanAns")
            print(ans)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return ans
    def run(self):
        header = ['ID', 'Number of Rounds', 'f_select_row','f_select_column','f_group_by','f_sort_by','f_add_column', '<END>']
        with open(self.anal_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)

        correct=0
        for i in range(207, len(self.train_dataset)):#len(self.train_dataset)
            data = self.train_dataset[i]
            T = data['table']
            Q = data['question']
            A = data['answers']
            id = data['id']
            A_hat, chain_dict, operations = self.chain_of_table(T,Q)
            # A_hat = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', A_hat)
            # if re.fullmatch(r'[\d,]+', A_hat):
            #     # Remove commas if it's a numeric string
            #     A_hat = A_hat.replace(',', '')      
            if A_hat==A[0]: correct+=1
            accuracy = correct/i
            if self.debug_acc or self.debug:
                print("#########################")
                print("Finale A", id,i)
                print(A)
                print("Finale A_hat", id,i)
                print(A_hat)
                print("Finale acc till now")
                print(accuracy)
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            row_anal=[]
            for operation in ['f_select_row','f_select_column','f_group_by','f_sort_by','f_add_column', '<END>']:
                if operation in chain_dict: val=chain_dict[operation]
                else: val='N/A'
                row_anal.append(val)
            with open(self.output_path, 'a', newline='') as tsvfile:
                # Create a CSV writer object with tab as a delimiter
                writer = csv.writer(tsvfile, delimiter='\t')
                writer.writerow([id, A_hat])
            with open(self.anal_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([id, len(operations)]+row_anal)
        print("TSV file created successfully with path "+ self.output_path)
        print("CSV file created successfully with path "+ self.anal_path)
        
            
    
    def prompt_vllm(self, prompt, llm, type, options_str=None):
        if type=="ans":
            #content = '''You are a helpful assistant. Provide only the direct answer in numeric form when the answer is a number. Do not write numbers as words (e.g., 4 instead of "four"). Provide only the final answer, with no extra words, explanations, or references to data structure (e.g., no mentioning of rows, columns, or other metadata). Your response should strictly be the final answer in lowercase, with no introductory phrases.'''
            #content = '''You are a helpful assistant. Provide only the direct answer, without any extra words, explanations, calculations, or references to data structure (e.g., no mentioning of rows, columns, or other metadata). Your response should be strictly the final answer in lowercase and if it's number, in numeric format, with no introductory phrases.'''
            content = "You are a helpful assistant. Provide only the direct answer, without any extra words, explanations, or references to data structure (e.g., no mentioning of rows, columns, or other metadata). Your response should be strictly the final answer in lowercase, with no introductory phrases."
        elif type=="f" and options_str is not None:
            ## Base prompt
            #content = "You are a helpful assistant who chooses the most suitable operations from the given operations "+options_str+". You give your answer at the end of your whole response in number. Answer strictly in number."
            ## Add think step by step
            content = "You are a helpful assistant who chooses the most suitable operations from the given operations "+options_str+". Think step by step. Give reasons first and then give your answer at the end of your whole response in number. Give your choice of operation strictly in number."
        else:
            content = "You are a helpful assistant who uses the examples provided to answer following the format."
        
        messages = [
        {"role": "system", "content": content},
        {"role": "user", "content": prompt},
        ]
        if self.debug:
            print("##########################")
            print("Content")
            print(content)
            print("Prompt")
            print(prompt)
            print("##########################")
        sampling_params = SamplingParams(max_tokens=500, min_tokens=0)
        #outputs = llm.generate(prompt, sampling_params)
        outputs = llm.chat(messages, sampling_params)
        output = outputs[0].outputs[0].text
        if self.debug:
            print("##########################")
            print("GO Start:")
            print(output)
            print("##########################")
        return output
    
    def prompt_model(self, prompt, model, tokenizer, device):
        # tokenizer = AutoTokenizer.from_pretrained(model_id)
        # model = AutoModelForCausalLM.from_pretrained(model_id)
        # model.to(device)


        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        #inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    

        outputs = model.generate(input_ids, do_sample=True, pad_token_id=tokenizer.eos_token_id) # max_length=100
        #print("is output tensor using gpu?", outputs.is_cuda)
        result=tokenizer.batch_decode(outputs, skip_special_tokens=True)
        prompt_len = len(prompt)
        if self.debug:
            print("##########################")
            print("OG output")
            print(result[0])
            print("##########################")
            print("GO START:")
        result = result[0][prompt_len:]
        if self.debug:
            print(result)
            print("##########################")
        return result
    def chain_to_string(self, chain):
        if self.debug:
            print("##########################")
            print("##########################")
            print("chain var in func")
            print(chain)
        chain_str = ""
        for operations in chain:
            if self.debug: print(operations)
            chain_str = chain_str + operations + " -> "
        if self.debug: print(chain_str)
        return chain_str
    

    def GenerateArgs(self,T, Q, f):
        '''
        T: pandas dataframe format
        '''
        if self.debug: print("GerARG f is", f)
        operations = ["f_add_column", "f_select_column", "f_select_row", "f_group_by", "f_sort_by"]
        # Get dataframe
        #df = pd.DataFrame(T['rows'], columns=T['header'])
        table_pipe = dfToPIPE(df=T)
        # Question or statement
        question = "Question: "+ Q
        statement = "statement : " + Q
        row_indices_list = T.index.tolist()
        row_num = len(row_indices_list)
        col_list = list(T.columns)
        col_num = len(col_list)
        if f == operations[0]:
            # Prompt
            gen_prompt = self.prompt_f_add_column + "\n" + table_pipe + "\n" + question
            prompt = self.prompt_f_add_column + "\n" + table_pipe + "\n" + question + "\n" + "Explanation: "
            # Generate column header
            prompt_header="Using "+ f +" to perform table operations to solve the above question regarding the table, what is the suitable column header for "+f+"? Respond with the column header that should be added and nothing else."
            prompt_header=gen_prompt+"\n"+prompt_header
            header="a string that is longer than 25 char"
            time=0
            while len(header)>25 and time<=5:
                header = self.prompt_vllm(prompt=prompt_header, llm=self.llm, type="arg")
                time+=1
            if len(header)>25: return None
            f_wHeader=f+'('+header+')'
            print("PRINT header", f_wHeader)
            prompt_arg = "Using "+ f_wHeader +" to perform table operations to solve the above question regarding the table, what is the suitable "+str(row_num)+" values to add the table for the new column "+"'header'"+""+"? Answer in the format: 'The value: value1 | value2 | ... | valueN'. There can only be "+str(row_num)+" values."
            prompt_arg = gen_prompt+"\n"+prompt_arg
            # Prompt model
            #raw_arg = self.prompt_model(prompt=prompt, model=self.model, tokenizer=self.tokenizer, device=self.device)
            values_str=None
            values=[]
            time=0
            while (values_str==None or len(values)!=row_num) and time<=5:
                raw_arg = self.prompt_vllm(prompt=prompt_arg, llm=self.llm, type="arg")
                pattern = "The value: (.*)"#"The value: (.*?)(?=\n/\*)"
                ##"The value: (.+?)(?=\n)"
                values_str = re.search(pattern, raw_arg, re.DOTALL) ##.group(1).replace("\n", "")
                print("PRINT VALUE STR", values_str)
                if not(values_str==None):
                    values_str = values_str.group(1).replace("\n", "")
                    values = re.split(r'\s*\|\s*', values_str) ##values_str.split(" | ")
                time+=1
                print("PRINT VALUES raw arg", raw_arg)
                print("PRINT VALUES", values)
            #if self.debug:
                # print("##########################")
                # print("GerArg prompt")
                # print(prompt)
                # print("##########################")
                # print("GENERATE ARGS raw_arg:")
                # print(raw_arg)
            # Regular Expression
            # pattern = "f_add_column\((.*?)\)"
            # column = re.search(pattern, raw_arg) ##.group(1)
            # if column==None: return None
            # else: column = column.group(1)
            if (values_str==None or len(values)!=row_num): return None
            
            
            # if self.debug:
            #     print("##########################")
            #     print("arg Column")
            #     print(column)
            #     print("##########################")
            #     print("arg values str")
            #     print(values_str)
            #     print("##########################")
            #     print("arg values list")
            #     print(values)
            # print("PRINT HEADER VALUE", header, values)
            print("PRINT HEADER return",header)
            print("PRINT values return", values)
            return [header, values]
            
        elif f == operations[1]: #f_select_column
            # Generate prompt
            gen_prompt = self.prompt_f_select_column + "\n" + table_pipe + "\n" + question
            col_ops=""
            for i,col in enumerate(col_list):
                col_ops=col_ops+"("+str(i+1)+")"+col+" "
            col_ops=col_ops.rstrip(" ")
            prompt_arg = "Using "+ f +" to perform table operations to solve the above question regarding the table, what are the suitable columns to choose from "+col_ops+"?  Choose the numbers associate with the columns that can best answer the question above. These are the options: "+col_ops+". Answer in the format: 'Columns to choose: number | number | ... | number'. Only respond with numbers."
            prompt_arg = gen_prompt+"\n"+prompt_arg
            # Prompt model
            #raw_arg = self.prompt_model(prompt=prompt, model=self.model, tokenizer=self.tokenizer, device=self.device)
            raw_arg = self.prompt_vllm(prompt=prompt_arg, llm=self.llm, type="arg")
            print("PRINT COL OPS", col_ops)
            matches=[]
            cols=[]
            time=0
            while (matches==[] or len(cols)>=col_num) and time<=5:
                raw_arg = self.prompt_vllm(prompt=prompt_arg, llm=llm, type="arg")
                matches = re.findall('(?<=Columns to choose: )(\d+(?:\s*\|\s*\d+)*)', raw_arg)
                print("PRINT RAW ARG", raw_arg)
                print("PRINT matches", matches)
                if not(matches==[]):
                    try:
                        cols = [int(num.strip()) for num in matches[0].split('|')]
                    except Exception as e:
                        matches=[]
                    for col in cols:
                        if col<1 or col>col_num: 
                            matches=[]
                            break
                print("PRINT COLS", cols)
                time+=1
            if (matches==[] or len(cols)>=col_num): return None
            arg_col_list = copy.deepcopy(col_list)
            keys=list(range(1,len(arg_col_list)+1))
            arg_dict=dict(zip(keys, arg_col_list))
            columns = [arg_dict[col] for col in cols]
            print("PRINT COLUMNS IN TXT", columns)
            # if self.debug:
            #     print("########################
            #     print("GerArg prompt")
            #     print(prompt)
            #     print("##########################")
            #     print("GENERATE ARGS raw_arg:")
            #     print(raw_arg)
            # Regular Expression
            # pattern = "f_select_column\((.*?)\)"
            # col_str = re.search(pattern, raw_arg)
            # if col_str == None: return None
            # else: columns = col_str.group(1)
            # if self.debug:
            #     print("##########################")
            #     print("GENERATE ARGS raw columns:")
            # columns = columns.lstrip("[").rstrip("]")
            # columns = re.split(r'\s*\,\s*', columns)

            # if self.debug:
            #     print("##########################")
            #     print("arg Columns")
            #     print(columns)
            return [columns]

            
        elif f == operations[2]: #f_select_row
            # Generate prompt
            gen_prompt = self.prompt_f_select_row + "\n" + table_pipe + "\n" + question
            prompt_arg = "Using "+ f +" to perform table operations to solve the above question regarding the table, what are the suitable rows to select from the table? Answer in the format: 'Rows to select: number | number | ... | number'."
            prompt_arg = gen_prompt+"\n"+prompt_arg
            
            # Prompt model
            #raw_arg = self.prompt_model(prompt=prompt, model=self.model, tokenizer=self.tokenizer, device=self.device)
            
            # if self.debug:
            #     print("##########################")
            #     print("GerArg prompt")
            #     print(prompt)
            #     print("##########################")
            #     print("GENERATE ARGS raw_arg:")
            #     print(raw_arg)
            matches=[]
            rows=[]
            time=0
            while (matches==[] or len(rows)>=row_num) and time<=5:
                raw_arg = self.prompt_vllm(prompt=prompt_arg, llm=llm, type="arg")
                matches = re.findall('(?<=Rows to select: )(\d+(?:\s*\|\s*\d+)*)', raw_arg)
                print("PRINT RAW ARG", raw_arg)
                print("PRINT matches", matches)
                if not(matches==[]):
                    try:
                        rows = [int(num.strip()) for num in matches[0].split('|')]
                    except Exception as e:
                        matches=[]
                print("PRINT ROWS", rows)
                time+=1
            if (matches==[] or len(rows)>=row_num): return None

            
            # Regular Expression
            # pattern = "f_select_row\((.*?)\)"
            # rows_str = re.search(pattern, raw_arg)
            # if rows_str == None: return None
            # else: rows = rows_str.group(1)
            # if self.debug:
            #     print("##########################")
            #     print("GENERATE ARGS raw columns:")
            #     print(rows)
            # rows = rows.lstrip("[").rstrip("]")
            # rows = re.split(r'\s*\,\s*', rows)
            # if self.debug:
            #     print("##########################")
            #     print("GENERATE ARGS rows:")
            #     print(rows)
            # rows = [row.replace(" ", "") for row in rows]
            # if self.debug:
            #     print("##########################")
            #     print("GENERATE ARGS new rows:")
            #     print(rows)
            
            # if self.debug: print("##########################")
            return [rows]
            
            
        elif f == operations[3]: #f_group_by
            # Generate prompt
            gen_prompt = self.prompt_f_group_by + "\n" + table_pipe + "\n" + question
            
            col_ops=""
            for i,col in enumerate(col_list):
                col_ops=col_ops+"("+str(i+1)+")"+col+" "
            col_ops=col_ops.rstrip(" ")
            prompt_arg = "Using "+ f +" to perform table operations to solve the above question regarding the table, what is the most suitable column to choose from "+col_ops+" to group the rows and get the corresponding count?  Choose the number associate with the columns that can best answer the question above. These are the options: "+col_ops+". Answer in the format: 'Column to choose: number'. Only respond with a number."
            prompt_arg = gen_prompt+"\n"+prompt_arg
            keys = [str(i) for i in range(1,len(col_list)+1)]
            match = None
            time=0
            print("PRINT COL LS", col_list)
            print("PRINT KEYS", keys)
            while (match==None or not(match.group(0) in keys)) and time<5:
                raw_f = self.prompt_vllm(prompt=prompt_arg, llm=self.llm, type="f")
                print("PRINT RAW F",raw_f)
                match = re.search(r"\d+$", raw_f)
                if not(match is None):
                    print("PRINT group COL", match.group(0))
                time+=1
            if (match==None or not(match.group(0) in keys)): return None
            arg_col_list = copy.deepcopy(col_list)

            arg_dict=dict(zip(keys, arg_col_list))
            column = arg_dict[match.group(0)]
            print("PRINT FINAL COLUMN", column)
            # Prompt model
            #raw_arg = self.prompt_model(prompt=prompt, model=self.model, tokenizer=self.tokenizer, device=self.device)
            # raw_arg = self.prompt_vllm(prompt=prompt_arg, llm=self.llm, type="arg")
            # if self.debug:
            #     print("##########################")
            #     print("GerArg prompt")
            #     print(prompt)
            #     print("##########################")
            #     print("GENERATE ARGS raw_arg:")
            #     print(raw_arg)
            # # Regular Expression
            # pattern = "f_group_by\((.*?)\)"
            # col_str = re.search(pattern, raw_arg)
            # if col_str == None: return None
            # else: column = col_str.group(1)
            # if self.debug:
            #     print("##########################")
            #     print("GENERATE ARGS raw column:")
            #     print(column)

            # if self.debug: print("##########################")
            return [column]
            
        elif f == operations[4]: #f_sort_by
            # Generate prompt
            gen_prompt = self.prompt_f_sort_by + "\n" + table_pipe + "\n" + question
            
            
            
            numeric_columns = list(T.select_dtypes(include=['number']).columns)
            if len(numeric_columns)==0: return None ##Will still be in one of the operations to choose from
            keys = [str(i) for i in range(1,len(numeric_columns)+1)]
            col_ops=""
            for i,col in enumerate(numeric_columns):
                col_ops=col_ops+"("+str(i+1)+")"+col+" "
            col_ops=col_ops.rstrip(" ")
            prompt_arg = "Using "+ f +" to perform table operations to solve the above question regarding the table, which is the most suitable column to choose from "+col_ops+" that can be used to sort the data?  Choose the numbers associate with the columns that can best answer the question above. These are the options: "+col_ops+". Answer in the format: 'Column to choose: number'. Only respond with a number."
            prompt_arg = gen_prompt+"\n"+prompt_arg
            # Prompt model
            #raw_arg = self.prompt_model(prompt=prompt, model=self.model, tokenizer=self.tokenizer, device=self.device)
            match = None
            time=0
            print("PRINT num COL LS", numeric_columns)
            print("PRINT KEYS", keys)
            while (match==None or not(match.group(0) in keys)) and time<5:
                raw_f = self.prompt_vllm(prompt=prompt_arg, llm=self.llm, type="f")
                print("PRINT RAW F",raw_f)
                match = re.search(r"\d+$", raw_f)
                if not(match is None):
                    print("PRINT sort COL", match.group(0))
                time+=1
            if (match==None or not(match.group(0) in keys)): return None
            
            arg_col_list = copy.deepcopy(numeric_columns)

            arg_dict=dict(zip(keys, arg_col_list))
            column = arg_dict[match.group(0)]
            print("PRINT sort COL", column)
            prompt_arg = "Using "+ f +" on the column '"+column+"' to sort table data to solve the above question, which is the most suitable order, (1)small to large or (2)large to small, to sort the table?  Choose the number associated with the option that's the best order to sort the table for answering the question above. These are the options: (1)small to large (2)large to small. Answer in the format: 'Order: number'. Only respond with a number."
            prompt_arg = gen_prompt+"\n"+prompt_arg
            match = None
            time=0
            while (match==None or not(match.group(0) in ['1','2'])) and time<5:
                raw_f = self.prompt_vllm(prompt=prompt_arg, llm=self.llm, type="f")
                print("PRINT RAW F",raw_f)
                match = re.search(r"\d+$", raw_f)
                if not(match is None):
                    print("PRINT sort ORDER num", match.group(0))
                time+=1
            if (match==None or not(match.group(0) in ['1','2'])): return None
            order_dict={'1':'Ascend','2':'Descend'}
            order = order_dict[match.group(0)]
            print("PRINT order str", order)
            # if self.debug:
            #     print("##########################")
            #     print("GerArg prompt")
            #     print(prompt)
            #     print("##########################")
            #     print("GENERATE ARGS raw_arg:")
            #     print(raw_arg)
            # # Regular Expression
            # pattern = "f_sort_by\((.*?)\)"
            # col_str = re.search(pattern, raw_arg) #the order is "small to large".
            # if col_str == None: return None
            # else: column = col_str.group(1)
            # pattern = "the order is \"(.*?)\""
            # order_str = re.search(pattern, raw_arg)
            # if order_str == None: return None
            # else: order_str = order_str.group(1)
            # if order_str == "small to large": order = "Ascend"
            # elif order_str == "large to small": order = "Descend"
            # else: return None
            # if self.debug:
            #     print("##########################")
            #     print("GENERATE ARGS raw column:")
            #     print(column)
            #     print("##########################")
            #     print("GENERATE ARGS order:")
            #     print(order_str)
            #     print("GENERATE ARGS order:")
            #     print(order)
            print("PRINT RETURN col order str", column, order)
            return [column, order]
    
    
    def extract_f(self, raw_f):
        # Build hash table for operations with their length
        operations = ["f_add_column", "f_select_column", "f_select_row", "f_group_by", "f_sort_by"]
        fhash={}
        operation_len=[]
        for operation in operations:
            operation_len.append(len(operation))
            key=len(operation)
            if key in fhash:
                key = str(key)
            fhash[key]=operation
        # fhash = {11:"f_add_col()", 17:"f_select_column()", 14:"f_select_row()", 
        #          12:"f_group_by()", "11":"f_sort_by()"} # f_sort_by has the same length so use "11" here
        #          #11, 17, 14, 12, 11; "f_add_col()", "f_select_column()", "f_select_row()", "f_group_by()", "f_sort_by()"
        print("Fhash", fhash)
        
        f=""
        for i in range(len(raw_f)):
            c = raw_f[i]
            print("->",i)
            if raw_f[i] == "f" and len(raw_f[i:])>2 and raw_f[i+1]=="_":
                # print(i,"length", len(raw_f[i:]))
                # print(len(raw_f[i:i+11]),"11",len(raw_f[i:i+17]), len(raw_f[i:i+14]), len(raw_f[i:i+12]))
                # print(raw_f[i:i+11])
                # print("11")
                # print(raw_f[i:i+17]) 
                # print(raw_f[i:i+14])
                # print(raw_f[i:i+12])
                # print('hash')
                # print(fhash[len(raw_f[i:i+14])])
                # print(fhash[len(raw_f[i:i+17])])
                # print(len(raw_f[i:])>14 ,raw_f[i:i+14]==fhash[len(raw_f[i:i+14])])
                if len(raw_f[i:])>operation_len[0] and raw_f[i:i+operation_len[0]]==operations[0]:
                    f = operations[0]#fhash[len(raw_f[i:i+operation_len[0]])]
       
                    print(i,'snipp',raw_f[:i+operation_len[0]])
                    break
                elif len(raw_f[i:])>operation_len[4] and raw_f[i:i+operation_len[4]]==operations[4]:
                    f = operations[4]#fhash[str(operation_len[-1])]

                    print(i,"snipp",raw_f[:i+operation_len[-1]])
                    break
                elif len(raw_f[i:])>operation_len[1] and raw_f[i:i+operation_len[1]]==operations[1]:
                    f = operations[1]#fhash[len(raw_f[i:i+operation_len[1]])]
    
                    print(i,"snipp",raw_f[:i+operation_len[1]])
                    break
                elif len(raw_f[i:])>operation_len[2] and raw_f[i:i+operation_len[2]]==operations[2]:
                    f = operations[2]#fhash[len(raw_f[i:i+operation_len[2]])]
             
                    print(i,'snipp',raw_f[:i+operation_len[2]])
                    break
                elif len(raw_f[i:])>operation_len[3] and raw_f[i:i+operation_len[3]]==operations[3]:
                    f = operations[3]#fhash[len(raw_f[i:i+operation_len[3]])]
                    
                    print(i,'snipp',raw_f[:i+operation_len[3]])
                    break
            elif raw_f[i] == "<" and len(raw_f[i:])>5 and raw_f[i+1]=="E" and raw_f[i+2]=="N" and raw_f[i+3]=="D" and raw_f[i+4]==">": #<END>
                f = "<END>"
                break
        
        return f
        # if f!="":
        #     chain.append(f)
        # return chain
    def f(self, df, f, args):
        
        operations = ["f_add_column", "f_select_column", "f_select_row", "f_group_by", "f_sort_by"]
        
        if f == operations[0]:
            return f_add_column(df=df, columnHeader=args[0], columnData=args[1])
            
            
        elif f == operations[1]:
            return f_select_column(df=df, colsToBeSelected=args[0])
        elif f == operations[2]:
            return f_select_row(df=df, rows_to_be_selected=args[0])
        elif f == operations[3]:
            return f_group_by(df=df, column=args[0])
        elif f == operations[4]:
            return f_sort_by(df=df, column=args[0], order=args[1])
    def operations2StrnDict(self, operations):
        
        allstr=""
        alldict={}
        for i,operation in enumerate(list(operations)):
            if allstr=="": allstr='('+str(i+1)+')'+operation+'()'
            else: allstr = allstr + ' or ' + '('+str(i+1)+')'+operation+'()'
            alldict[str(i+1)]=operation
        return allstr,alldict
   
    def DynamicPlan(self, T, Q, chain, operations):
        '''
        T: pandas dataframe format
        '''
        examples='If the table needs more columns to answer the question, we can first use f_add_column() to\n'+\
            'add more columns to the table.\n'+\
            '/*\n'+\
            'col : Week | When | Kickoff | Opponent | Results; Final score | Results; Team record\n'+\
            'row 1 : 1 | Saturday, April 13 | 7:00 p.m. | at Rhein Fire | W 27-21 | 1-0\n'+\
            'row 2 : 2 | Saturday, April 20 | 7:00 p.m. | London Monarchs | W 37-3 | 2-0\n'+\
            'row 3 : 3 | Sunday, April 28 | 6:00 p.m. | at Barcelona Dragons | W 33-29 | 3-0\n'+\
            '*/\n'+\
            'Question: what is the date of the competition with highest attendance?\n'+\
            'The existing columns are: "Week", "When", "Kickoff", "Opponent", "Results; Final score",\n'+\
            '"Results; Team record", "Game site", "Attendance".\n'+\
            'Explanation: the question asks about the date of the competition with highest score. Each\n'+\
            'row is about one competition. We extract the value from column "Attendance" and create a\n'+\
            'different column "Attendance number" for each row. The datatype is Numerical.\n'+\
            'Therefore, the answer is: f_add_column(Attendance number). The value: 32092 | 34186 | 17503\n'+\
            "\n"+\
            "If the table only needs a few rows to answer the question, we use f_select_row() to select\n"+\
            "these rows for it. For example,\n"+\
            "/*\n"+\
            "col : Home team | Home Team Score | Away Team | Away Team Score | Venue | Crowd\n"+\
            "row 1 : st kilda | 13.12 (90) | melbourne | 13.11 (89) | moorabbin oval | 18836\n"+\
            "row 2 : south melbourne | 9.12 (66) | footscray | 11.13 (79) | lake oval | 9154\n"+\
            "row 3 : richmond | 20.17 (137) | fitzroy | 13.22 (100) | mcg | 27651\n"+\
            "*/\n"+\
            "Question : Whose home team score is higher, richmond or st kilda?\n"+\
            "Function: f_select_row(row 1, row 3)\n"+\
            "Explanation: The question asks about the home team score of richmond and st kilda. We need\n"+\
            "to know the the information of richmond and st kilda in row 1 and row 3. We select row 1\n"+\
            "and row 3.\n"+\
            "\n"+\
            "If the table only needs a few columns to answer the question, we use\n"+\
            "f_select_column() to select these columns for it. For example,\n"+\
            '/*\n'+\
            'table_caption : south wales derby\n'+\
            'col : competition | total matches | cardiff win | draw | swansea win\n'+\
            'row 1 : league | 55 | 19 | 16 | 20\n'+\
            'row 2 : fa cup matches | 2 | 0 | 27 | 2\n'+\
            'row 3 : league cup | 5 | 2 | 0 | 3\n'+\
            '*/\n'+\
            'Question: what is the competition with the highest number of total matches?\n'+\
            'Explanation: the question wants to know what competition has the highest number of total matches.\n'+\
            'We need to know each competition\'s total matches so we extract "competition" and "total matches".\n'+\
            'The answer is : f_select_column([competition, total matches])\n'+\
            "\n"+\
            "If the question asks about items with the same value and the number of these items, we use\n"+\
            "f_group_by() to group the items. For example,\n"+\
            '/*\n'+\
            'col : Rank | Lane | Athlete | Time | Country\n'+\
            'row 1 : 1 | 6 | Manjeet Kaur (IND) | 52.17 | IND\n'+\
            'row 2 : 2 | 5 | Olga Tereshkova (KAZ) | 51.86 | KAZ\n'+\
            'row 3 : 3 | 4 | Pinki Pramanik (IND) | 53.06 | IND\n'+\
            'row 4 : 4 | 1 | Tang Xiaoyin (CHN) | 53.66 | CHN\n'+\
            'row 5 : 5 | 8 | Marina Maslyonko (KAZ) | 53.99 | KAZ\n'+\
            '*/\n'+\
            'Question: tell me the number of athletes from japan.\n'+\
            'The existing columns are: Rank, Lane, Athlete, Time, Country.\n'+\
            'Explanation: The question asks about the number of athletes from India. Each row is about\n'+\
            'an athlete. We can group column "Country" to group the athletes from the same country.\n'+\
            'Therefore, the answer is: f_group_by(Country).\n'+\
            "\n"+\
            "If the question asks about the order of items in a column, we use f_sort_by() to sort\n"+\
            "the items. For example,\n"+\
            '/*\n'+\
            'col : Position | Club | Played | Points | Wins | Draws | Losses | Goals for | Goals against\n'+\
            'row 1 : 1 | Malaga CF | 42 | 79 | 22 | 13 | 7 | 72 | 47\n'+\
            'row 10 : 10 | CP Merida | 42 | 59 | 15 | 14 | 13 | 48 | 41\n'+\
            'row 3 : 3 | CD Numancia | 42 | 73 | 21 | 10 | 11 | 68 | 40\n'+\
            '*/\n'+\
            'Question: what club placed in the last position?\n'+\
            'The existing columns are: Position, Club, Played, Points, Wins, Draws, Losses, Goals for,\n'+\
            'Goals against\n'+\
            'Explanation: the question asks about the club in the last position. Each row is about a\n'+\
            'club. We need to know the order of position from last to front. There is a column for\n'+\
            'position and the column name is Position. The datatype is Numerical.\n'+\
            'Therefore, the answer is: f_sort_by(Position), the order is "large to small".'
            # "\n"+\
            # "Here are examples of using the operations to answer the question.\n"+\
            # "/*\n"+\
            # "col : Date | Division | League | Regular Season | Playoffs | Open Cup\n"+\
            # "row 1 : 2001/01/02 | 2 | USL A-League | 4th, Western | Quarterfinals | Did not qualify\n"+\
            # "row 2 : 2002/08/06 | 2 | USL A-League | 2nd, Pacific | 1st Round | Did not qualify\n"+\
            # "row 5 : 2005/03/24 | 2 | USL First Division | 5th | Quarterfinals | 4th Round\n"+\
            # "*/\n"+\
            # "Question: what was the last year where this team was a part of the usl a-league?\n"+\
            # "Function Chain: f_add_column(Year) -> f_select_row(row 1, row 2) ->\n"+\
            # "f_select_column(Year, League) -> f_sort_by(Year) -> <END>\n"+\
            # "......"
        # Get dataframe
        #df = pd.DataFrame(T['rows'], columns=T['header'])
        all_operations = set(['f_select_row','f_select_column','f_group_by','f_sort_by','f_add_column', '<END>'])
        rest_operations = all_operations - operations
        print("##########")
        print(T)
        print("$$$$$$$$$$$$$$$$")
        print("PRINT rest of operations", rest_operations)
        rest_oper_str,hash_f = self.operations2StrnDict(rest_operations)
        table_pipe = dfToPIPE(df=T)
        # Question
        question = "Question: "+ Q
        # Options "f_add_col()", "f_select_column()", "f_select_row()", "f_group_by()", "f_sort_by()"
        options = "The next operation must be one of " + rest_oper_str 
        # Chain
        print("###################")
        print("Chain variable")
        print(chain)
        if len(chain)>0: chain_str = "Operations so far: " + self.chain_to_string(chain=chain)
        else: chain_str = "What should be the first operation?"
        print("###################")
        print("OLD chain")
        print(chain_str)
        # Generate prompt 
        prompt = examples + "\n" + table_pipe + "\n" + question + "\n" + chain_str + "\n" + options
        # Prompt model
        #raw_f = self.prompt_model(prompt=prompt, model=self.model, tokenizer=self.tokenizer, device=self.device)
        #raw_f = self.prompt_vllm(prompt=prompt, llm=self.llm, type="f")
        
        #hash_f={'1':'f_select_row','2':'f_select_column','3':'f_group_by','4':'f_sort_by','5':'f_add_column', '6':'<END>'}
        match = None
        time=0
        while (match==None or not(match.group(1) in hash_f)) and time<5:
            raw_f = self.prompt_vllm(prompt=prompt, llm=self.llm, type="f", options_str=rest_oper_str )
            print("PRINT RAW F",raw_f)
            match = re.search(r'(\d+)$', raw_f)
            time+=1
        
        # Extract operation from model output and append to chain
        #chain = self.extract_f(raw_f=raw_f)
        if (match==None or not(match.group(1) in hash_f)): f=""
        else: f=hash_f[match.group(1)]
        #f = self.extract_f(raw_f=raw_f)
        print("PRINT RAW F", raw_f, "FInal f", f)
        return f
        

if __name__=="__main__":
    ## Check if we have cuda?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, "; using gpu:", torch.cuda.get_device_name())
    
    ## Load model
    # model_id = "meta-llama/Meta-Llama-3-8B"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id)
    # model.to(device)

    ## Load dataset
    trainDataset, valDataset, testDataset = dataset() #id, question, answers, table
    print(trainDataset)
    print(valDataset)
    print(testDataset)
    print("FIRST data instance")
    print(trainDataset[0])
    print("##################################")


    ## Load model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    #model_id = "meta-llama/Llama-2-13b-chat-hf"
    llm = LLM(model=model_id, device=device)

    ## Output file path
    now = datetime.now()
    # Format the datetime as numeric only: YYYYMMDDHHMMSS
    numeric_datetime = now.strftime("%Y%m%d%H%M%S")
    output_path = os.getcwd()+"/wikitq_out/" + numeric_datetime + ".tsv"
    anal_path = os.getcwd()+"/wikitq_eval_out/" + numeric_datetime + ".csv"
    print("Output file path:", output_path) 
    print("Analysis file path:", anal_path)    

    
    chain_of_table = ChainOfTable(train_dataset=trainDataset, llm=llm, model="model", 
                                    tokenizer="tok",device=device, output_path=output_path, anal_path=anal_path, debug=True, debug_acc=True)
    chain_of_table.run()

    
    
