from datasets import load_dataset
import torch
from vllm import LLM, SamplingParams

def dataset():
    dataset = load_dataset("Stanford/wikitablequestions")
    trainDataset = dataset['train']
    valDataset = dataset['validation']
    testDataset  = dataset['test']
    return trainDataset, valDataset, testDataset


def prompt_vllm(self, prompt, llm, type, options_str=None):

    if type=="ans":
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

if __name__=='__main__':
    ## Check if we have cuda?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using", device, "; using gpu:", torch.cuda.get_device_name())

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
    llm = LLM(model=model_id, device=device)
    
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    content = "You are a helpful assistant. Provide only the direct answer, without any extra words, explanations, or references to data structure (e.g., no mentioning of rows, columns, or other metadata). Your response should be strictly the final answer in lowercase, with no introductory phrases."
    

    messages = [[
        {"role": "system", "content": content},

        {"role": "user", "content": "Hello, my name is"},

    ],
    [
        {"role": "system", "content": content},
        {"role": "user", "content": "The capital of France is"},
    ]]
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
    for output in outputs:
        generated_text = output.outputs[0].text
        print(generated_text)
    # output = outputs[0].outputs[0].text