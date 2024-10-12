from datetime import datetime
import csv
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
import transformers
import torch
import re

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
    '''
    try:
        rowsToBeSelected = rows_to_be_selected.copy()
        for i in range(len(rowsToBeSelected)):
            row = rowsToBeSelected[i]
            ## Error prone
            row = int(row[3:]) - 1
            rowsToBeSelected[i] = row
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
    
    # oneTable=trainDataset[0]['table']
    # df = pd.DataFrame(oneTable['rows'], columns=oneTable['header'])
    
    
    
    # List of values
    # new_column_data = ['Value1', 'Value2', 'Value3', 'Value4', 'Value5', 'Value6', 'Value7', 'Value8', 'Value9', 'Value10']
    # df=f_add_col(df,"New Data",new_column_data)
    # df=f_select_column(df, ["Year","New Data"])
    # df = f_select_row(df,["row 1", "row 2", "row 3", "row 4", "row 5", "row 6"] )
    # df = f_group_by(df, "League")
    # df = f_sort_by(df, "Count", "Ascend")
    # print(df)
    # print("----------------------------")
    # strPIPE = dfToPIPE(df)
    # print(strPIPE)

class LoadModel:
    def __init__(self, pretrained_model, quantization, device):
        self.quantization=quantization
        self.pretrained_model=pretrained_model
        self.device=device
    def load_model(self):
        if self.quantization == "8bit":
            config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf4",
                bnb_8bit_use_double_quant=True,
                bnb_8bit_compute_dtype=torch.bfloat16,
            )
        elif self.quantization == "4bit":
            config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        if self.quantization == "no_quant":
            model = AutoModelForCausalLM.from_pretrained(self.pretrained_model)
        else:   
            model = AutoModelForCausalLM.from_pretrained(
                self.pretrained_model,
                quantization_config=config,
                device_map=self.device
            )
        return model


# def prompt_model(prompt, model_id, device):
#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id)
#     model.to(device)


#     input_ids = tokenizer(prompt, return_tensors="pt").input_ids
#     input_ids = input_ids.to(device)
#     #inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    

#     outputs = model.generate(input_ids, do_sample=True, max_length=30, pad_token_id=tokenizer.eos_token_id)
#     print("is output tensor using gpu?", outputs.is_cuda)
#     print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# def chain_to_string(chain):
#     chain_str = ""
#     for operations in chain:
#         chain_str + " -> " + operations
#     return chain_str
# def DynamicPlan(T, Q, chain):
#     examples="If the table only needs a few rows to answer the question, we use f_select_row() to select\n"+\
#         "these rows for it. For example,\n"+\
#         "/*\n"+\
#         "col : Home team | Home Team Score | Away Team | Away Team Score | Venue | Crowd\n"+\
#         "row 1 : st kilda | 13.12 (90) | melbourne | 13.11 (89) | moorabbin oval | 18836\n"+\
#         "row 2 : south melbourne | 9.12 (66) | footscray | 11.13 (79) | lake oval | 9154\n"+\
#         "row 3 : richmond | 20.17 (137) | fitzroy | 13.22 (100) | mcg | 27651\n"+\
#         "*/\n"+\
#         "Question : Whose home team score is higher, richmond or st kilda?\n"+\
#         "Function: f_select_row(row 1, row 3)\n"+\
#         "Explanation: The question asks about the home team score of richmond and st kilda. We need\n"+\
#         "to know the the information of richmond and st kilda in row 1 and row 3. We select row 1\n"+\
#         "and row 3.\n"+\
#         "If the table only needs a few columns to answer the question, we use\n"+\
#         "f_select_column() to select these columns for it. For example,\n"+\
#         "......\n"+\
#         "If the question asks about items with the same value and the number of these items, we use\n"+\
#         "f_group_by() to group the items. For example,\n"+\
#         "......\n"+\
#         "If the question asks about the order of items in a column, we use f_sort_by() to sort\n"+\
#         "the items. For example,\n"+\
#         "......\n"+\
#         "Here are examples of using the operations to answer the question.\n"+\
#         "/*\n"+\
#         "col : Date | Division | League | Regular Season | Playoffs | Open Cup\n"+\
#         "row 1 : 2001/01/02 | 2 | USL A-League | 4th, Western | Quarterfinals | Did not qualify\n"+\
#         "row 2 : 2002/08/06 | 2 | USL A-League | 2nd, Pacific | 1st Round | Did not qualify\n"+\
#         "row 5 : 2005/03/24 | 2 | USL First Division | 5th | Quarterfinals | 4th Round\n"+\
#         "*/\n"+\
#         "Question: what was the last year where this team was a part of the usl a-league?\n"+\
#         "Function Chain: f_add_column(Year) -> f_select_row(row 1, row 2) ->\n"+\
#         "f_select_column(Year, League) -> f_sort_by(Year) -> <END>\n"+\
#         "......"
#     # Get dataframe
#     df = pd.DataFrame(T['rows'], columns=T['header'])
#     table_pipe = dfToPIPE(df=df)
#     # Question
#     question = "Question: "+ Q
#     # Options
#     options = "The next operation must be one of f_select_row() or f_select_column() or f_group_by()\n"+\
#                 "or f_sort_by()."
#     # Chain
#     chain_str = "Function Chain: " + chain_to_string(chain=chain)
#     # Prompt
#     prompt = examples + "\n" + table_pipe + "\n" + question + "\n" + options + "\n" + chain_str 
#     print(prompt)

class ChainOfTable:
    def __init__(self, train_dataset, llm, model, tokenizer, device, output_path):
        self.output_path = output_path
        self.llm=llm
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
        # Get dataframe
        df = pd.DataFrame(T['rows'], columns=T['header'])
        round=0
        while True:
            print("Round", round, "Chain", chain)
            f = self.DynamicPlan(T=df, Q=Q, chain=chain)
            if f=="": 
                print("f is empty")
                break
            elif f=="<END>": 
                print("#########################")
                print('LAST end f',f)
                chain.append(f)
                break
            #chain = ["f_sort_by"]
            print("#########################")
            print('new f',f)
            print("#########################")
            args = self.GenerateArgs(T=df, Q=Q, f=f)
            if args==None: 
                print("args is None")
                break
            print("#########################")
            print("OG ARGSS")
            print(args)
            # Update table
            print("#########################")
            print("OLD DF")
            print(df)
            old_df = df
            df = self.f(df=df, f=f, args=args)
            if not(isinstance(df, pd.DataFrame)):
                print("df is gibberish")
                df = old_df
                break
            print("NEW DF")
            print(df)
            print("#########################")
            print("check type")
            print(args)
            if isinstance(args[0], list): argument = ', '.join(args[0])
            else: argument = args[0]
            f=f+"("+argument+")"
            print("#########Old chain################")
            print(chain)
            chain.append(f)

            print("#########Append chain################")
            print(chain)
            round+=1
            
        
        raw_a=self.Query(T=df, Q=Q, chain=chain, original_table=T)
        pattern = "^(.*)"
        a = re.search(pattern, raw_a) ##.group(1)
        if a==None: return ""
        else: 
            a = a.group(1)
            a = a.strip(" ")
            return a

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
        print("#########################")
        print("RawAns")
        print(raw_answer)
        ans = raw_answer.strip(' ').strip('\n')
        print("CleanAns")
        print(ans)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        return ans
    def run(self):
        with open(self.output_path, 'w', newline='') as tsvfile:
                # Create a CSV writer object with tab as a delimiter
            writer = csv.writer(tsvfile, delimiter='\t')

            correct=0
            for i in range(1,len(self.train_dataset)):
                data = self.train_dataset[i]
                T = data['table']
                Q = data['question']
                A = data['answers']
                id = data['id']
                A_hat = self.chain_of_table(T,Q)
                # A_hat = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', A_hat)
                # if re.fullmatch(r'[\d,]+', A_hat):
                #     # Remove commas if it's a numeric string
                #     A_hat = A_hat.replace(',', '')      
                if A_hat==A[0]: correct+=1
                accuracy = correct/i
                print("#########################")
                print("Finale A", id,i)
                print(A)
                print("Finale A_hat", id,i)
                print(A_hat)
                print("Finale acc till now")
                print(accuracy)
                print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
                writer.writerow([id, A_hat])
        print("TSV file created successfully with path "+ self.output_path)
            
    
    def prompt_vllm(self, prompt, llm, type):
        if type=="ans":
            #content = '''You are a helpful assistant. Provide only the direct answer in numeric form when the answer is a number. Do not write numbers as words (e.g., 4 instead of "four"). Provide only the final answer, with no extra words, explanations, or references to data structure (e.g., no mentioning of rows, columns, or other metadata). Your response should strictly be the final answer in lowercase, with no introductory phrases.'''
            #content = '''You are a helpful assistant. Provide only the direct answer, without any extra words, explanations, calculations, or references to data structure (e.g., no mentioning of rows, columns, or other metadata). Your response should be strictly the final answer in lowercase and if it's number, in numeric format, with no introductory phrases.'''
            content = "You are a helpful assistant. Provide only the direct answer, without any extra words, explanations, or references to data structure (e.g., no mentioning of rows, columns, or other metadata). Your response should be strictly the final answer in lowercase, with no introductory phrases."
        else:
            content = "You are a helpful assistant who uses the examples provided to answer following the format."
        
        messages = [
        {"role": "system", "content": content},
        {"role": "user", "content": prompt},
        ]
        print("##########################")
        print("Content")
        print(content)
        print("Prompt")
        print(prompt)
        print("##########################")
        sampling_params = SamplingParams(max_tokens=200, min_tokens=0)
        #outputs = llm.generate(prompt, sampling_params)
        outputs = llm.chat(messages, sampling_params)
        output = outputs[0].outputs[0].text
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
        print("##########################")
        print("OG output")
        print(result[0])
        # print("##########################")
        print("GO START:")
        result = result[0][prompt_len:]
        print(result)
        print("##########################")
        return result
    def chain_to_string(self, chain):
        print("##########################")
        print("##########################")
        print("chain var in func")
        print(chain)
        chain_str = ""
        for operations in chain:
            print(operations)
            chain_str = chain_str + operations + " -> "
        print(chain_str)
        return chain_str
    
    # def regular_expression(self, pattern, text):
    #     print()
    #     match = re.search(pattern, text)
    #     return match.group(1)
    def GenerateArgs(self,T, Q, f):
        '''
        T: pandas dataframe format
        '''
        print("GerARG f is", f)
        operations = ["f_add_column", "f_select_column", "f_select_row", "f_group_by", "f_sort_by"]
        # Get dataframe
        #df = pd.DataFrame(T['rows'], columns=T['header'])
        table_pipe = dfToPIPE(df=T)
        # Question or statement
        question = "Question: "+ Q
        statement = "statement : " + Q
        if f == operations[0]:
            # Generate prompt
            prompt = self.prompt_f_add_column + "\n" + table_pipe + "\n" + question + "\n" + "Explanation: "
            # Prompt model
            #raw_arg = self.prompt_model(prompt=prompt, model=self.model, tokenizer=self.tokenizer, device=self.device)
            raw_arg = self.prompt_vllm(prompt=prompt, llm=self.llm, type="arg")
            print("##########################")
            print("GerArg prompt")
            print(prompt)
            print("##########################")
            print("GENERATE ARGS raw_arg:")
            print(raw_arg)
            # Regular Expression
            pattern = "f_add_column\((.*?)\)"
            column = re.search(pattern, raw_arg) ##.group(1)
            if column==None: return None
            else: column = column.group(1)
            
            pattern = "The value: (.*?)(?=\n/\*)"
            ##"The value: (.+?)(?=\n)"
            values_str = re.search(pattern, raw_arg, re.DOTALL) ##.group(1).replace("\n", "")
            if values_str==None: return None
            else: values_str = values_str.group(1).replace("\n", "")
            values = re.split(r'\s*\|\s*', values_str) ##values_str.split(" | ")
            print("##########################")
            print("arg Column")
            print(column)
            print("##########################")
            print("arg values str")
            print(values_str)
            print("##########################")
            print("arg values list")
            print(values)
            return [column, values]
            
        elif f == operations[1]: #f_select_column
            # Generate prompt
            prompt = self.prompt_f_select_column + "\n" + table_pipe + "\n" + question
            # Prompt model
            #raw_arg = self.prompt_model(prompt=prompt, model=self.model, tokenizer=self.tokenizer, device=self.device)
            raw_arg = self.prompt_vllm(prompt=prompt, llm=self.llm, type="arg")
            print("##########################")
            print("GerArg prompt")
            print(prompt)
            print("##########################")
            print("GENERATE ARGS raw_arg:")
            print(raw_arg)
            # Regular Expression
            pattern = "f_select_column\((.*?)\)"
            col_str = re.search(pattern, raw_arg)
            if col_str == None: return None
            else: columns = col_str.group(1)
            print("##########################")
            print("GENERATE ARGS raw columns:")
            columns = columns.lstrip("[").rstrip("]")
            columns = re.split(r'\s*\,\s*', columns)
            # pattern = "The value: (.*?)(?=\n/\*)"
            # ##"The value: (.+?)(?=\n)"
            # values_str = re.search(pattern, raw_arg, re.DOTALL).group(1).replace("\n", "")
            # values = re.split(r'\s*\|\s*', values_str)##values_str.split(" | ")
            print("##########################")
            print("arg Columns")
            print(columns)
            return [columns]
            # print("##########################")
            # print("arg values str")
            # print(values_str)
            # print("##########################")
            # print("arg values list")
            # print(values)
            
        elif f == operations[2]: #f_select_row
            # Generate prompt
            prompt = self.prompt_f_select_row + "\n" + table_pipe + "\n" + question
            # Prompt model
            #raw_arg = self.prompt_model(prompt=prompt, model=self.model, tokenizer=self.tokenizer, device=self.device)
            raw_arg = self.prompt_vllm(prompt=prompt, llm=llm, type="arg")
            print("##########################")
            print("GerArg prompt")
            print(prompt)
            print("##########################")
            print("GENERATE ARGS raw_arg:")
            print(raw_arg)
            # Regular Expression
            pattern = "f_select_row\((.*?)\)"
            rows_str = re.search(pattern, raw_arg)
            if rows_str == None: return None
            else: rows = rows_str.group(1)
            print("##########################")
            print("GENERATE ARGS raw columns:")
            print(rows)
            rows = rows.lstrip("[").rstrip("]")
            rows = re.split(r'\s*\,\s*', rows)
            print("##########################")
            print("GENERATE ARGS rows:")
            print(rows)
            rows = [row.replace(" ", "") for row in rows]
            print("##########################")
            print("GENERATE ARGS new rows:")
            print(rows)
            # pattern = "The value: (.*?)(?=\n/\*)"
            # ##"The value: (.+?)(?=\n)"
            # values_str = re.search(pattern, raw_arg, re.DOTALL).group(1).replace("\n", "")
            # values = re.split(r'\s*\|\s*', values_str)##values_str.split(" | ")
            print("##########################")
            return [rows]
            
            
        elif f == operations[3]: #f_group_by
            # Generate prompt
            prompt = self.prompt_f_group_by + "\n" + table_pipe + "\n" + question
            # Prompt model
            #raw_arg = self.prompt_model(prompt=prompt, model=self.model, tokenizer=self.tokenizer, device=self.device)
            raw_arg = self.prompt_vllm(prompt=prompt, llm=self.llm, type="arg")
            print("##########################")
            print("GerArg prompt")
            print(prompt)
            print("##########################")
            print("GENERATE ARGS raw_arg:")
            print(raw_arg)
            # Regular Expression
            pattern = "f_group_by\((.*?)\)"
            col_str = re.search(pattern, raw_arg)
            if col_str == None: return None
            else: column = col_str.group(1)
            print("##########################")
            print("GENERATE ARGS raw column:")
            print(column)
            # rows = rows.lstrip("[").rstrip("]")
            # rows = re.split(r'\s*\,\s*', rows)
            # print("##########################")
            # print("GENERATE ARGS rows:")
            # print(rows)
            # rows = [row.replace(" ", "") for row in rows]
            # print("##########################")
            # print("GENERATE ARGS new rows:")
            # print(rows)
            # pattern = "The value: (.*?)(?=\n/\*)"
            # ##"The value: (.+?)(?=\n)"
            # values_str = re.search(pattern, raw_arg, re.DOTALL).group(1).replace("\n", "")
            # values = re.split(r'\s*\|\s*', values_str)##values_str.split(" | ")
            print("##########################")
            return [column]
            
        elif f == operations[4]: #f_sort_by
            # Generate prompt
            prompt = self.prompt_f_sort_by + "\n" + table_pipe + "\n" + question
            # Prompt model
            #raw_arg = self.prompt_model(prompt=prompt, model=self.model, tokenizer=self.tokenizer, device=self.device)
            raw_arg = self.prompt_vllm(prompt=prompt, llm=self.llm, type="arg")
            print("##########################")
            print("GerArg prompt")
            print(prompt)
            print("##########################")
            print("GENERATE ARGS raw_arg:")
            print(raw_arg)
            # Regular Expression
            pattern = "f_sort_by\((.*?)\)"
            col_str = re.search(pattern, raw_arg) #the order is "small to large".
            if col_str == None: return None
            else: column = col_str.group(1)
            pattern = "the order is \"(.*?)\""
            ##"The value: (.+?)(?=\n)"
            order_str = re.search(pattern, raw_arg)
            if order_str == None: return None
            else: order_str = order_str.group(1)
            if order_str == "small to large": order = "Ascend"
            elif order_str == "large to small": order = "Descend"
            else: return None
            print("##########################")
            print("GENERATE ARGS raw column:")
            print(column)
            print("##########################")
            print("GENERATE ARGS order:")
            print(order_str)
            print("GENERATE ARGS order:")
            print(order)
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
   
    def DynamicPlan(self, T, Q, chain):
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
            'Therefore, the answer is: f_sort_by(Position), the order is "large to small".\n'+\
            "\n"+\
            "Here are examples of using the operations to answer the question.\n"+\
            "/*\n"+\
            "col : Date | Division | League | Regular Season | Playoffs | Open Cup\n"+\
            "row 1 : 2001/01/02 | 2 | USL A-League | 4th, Western | Quarterfinals | Did not qualify\n"+\
            "row 2 : 2002/08/06 | 2 | USL A-League | 2nd, Pacific | 1st Round | Did not qualify\n"+\
            "row 5 : 2005/03/24 | 2 | USL First Division | 5th | Quarterfinals | 4th Round\n"+\
            "*/\n"+\
            "Question: what was the last year where this team was a part of the usl a-league?\n"+\
            "Function Chain: f_add_column(Year) -> f_select_row(row 1, row 2) ->\n"+\
            "f_select_column(Year, League) -> f_sort_by(Year) -> <END>\n"+\
            "......"
        # Get dataframe
        #df = pd.DataFrame(T['rows'], columns=T['header'])
        print("##########")
        print(T)
        print("$$$$$$$$$$$$$$$$")
        table_pipe = dfToPIPE(df=T)
        # Question
        question = "Question: "+ Q
        # Options "f_add_col()", "f_select_column()", "f_select_row()", "f_group_by()", "f_sort_by()"
        options = "The next operation must be one of f_select_row() or f_select_column() or f_group_by()\n"+\
                    "or f_sort_by() or f_add_column()."
        # Chain
        print("###################")
        print("Chain variable")
        print(chain)
        chain_str = "Function Chain: " + self.chain_to_string(chain=chain)
        print("###################")
        print("OLD chain")
        print(chain_str)
        # Generate prompt 
        prompt = examples + "\n" + table_pipe + "\n" + question + "\n" + options + "\n" + chain_str
        # Prompt model
        #raw_f = self.prompt_model(prompt=prompt, model=self.model, tokenizer=self.tokenizer, device=self.device)
        raw_f = self.prompt_vllm(prompt=prompt, llm=self.llm, type="f")
        # Extract operation from model output and append to chain
        #chain = self.extract_f(raw_f=raw_f)
        f = self.extract_f(raw_f=raw_f)
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
    llm = LLM(model=model_id)

    ## Output file path
    now = datetime.now()
    # Format the datetime as numeric only: YYYYMMDDHHMMSS
    numeric_datetime = now.strftime("%Y%m%d%H%M%S")
    output_path = "/home/pwang71/withKoehn/table_qa/wikitq_out/" + numeric_datetime + ".tsv"
    print("Output file path:", output_path)    

    
    chain_of_table = ChainOfTable(train_dataset=trainDataset, llm=llm, model="model", tokenizer="tok",device=device, output_path=output_path)
    chain_of_table.run()

    
    
