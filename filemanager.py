import csv
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def fullcombine(predpath,csvpath,outpath):
    rounddict={}
    ## Read csv and get ids and rounds
    with open(csvpath, "r") as infile:
        reader = csv.reader(infile)
        next(reader)  # Skip header
        reader = list(reader)
        # Write each row line by line
        for row in reader:
            rounddict[row[0]]=row[1] #key is ID, value is round
    print("CSV:", csvpath)
    print(len(rounddict),"data entries has been read from CSV")
    
    ## Read prediction from tsv and get ids and prediction results
    # True->1, False->
    preddict={}
    truecount=0
    with open(predpath, "r") as infile:
        for line in infile:
            line_ls=line.split('\t')
            if line_ls[1]=='True': preddict[line_ls[0]]=1
            else: preddict[line_ls[0]]=0
            truecount+=preddict[line_ls[0]]
    print("Pred TSV:", predpath)
    print(len(preddict),"data entries has been read from pred TSV")
    print("Acc from this file:", truecount/len(preddict))
    
    ## Write to outpath
    header = ['ID', 'Prediction','Round']
    # Write the header to the CSV file (only once)
    with open(outpath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

    # Append rows in each loop iteration
    for id,pred in preddict.items():  # Example loop, replace with your actual loop
        round=rounddict[id]

        # Open the file in append mode and write one row
        with open(outpath, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([id, pred, round])
    print('Output file generated at path:', outpath)
    ## Verify the output file
    outn=0
    with open(outpath, "r") as infile:
        for line in infile:
            outn+=1
    print("Output file length(including header):", outn)
            

def csvcombine(csvdirpath, filenames, outpath):
    tol=0
    # Open the output file and write header + data rows from all files
    with open(outpath, "w", newline="") as outfile:
        writer = csv.writer(outfile)
        header_written = False
        
        for filename in filenames:
            with open(csvdirpath+filename+'.csv', "r") as infile:
                reader = csv.reader(infile)
                # Write the header once
                if not header_written:
                    writer.writerow(next(reader))  # Write header from the first file
                    header_written = True
                else:
                    next(reader)  # Skip header for subsequent files
                reader = list(reader)
                flen = len(reader)
                tol+=flen
                fstartid = reader[0][0]
                fendid = reader[-1][0]
                print('>>'+filename+'.csv<<'+',Length:',flen,', Start ID:', fstartid, ', End ID:', fendid)
                # Write each row line by line
                for row in reader:
                    writer.writerow(row)
    print('Combined data length:', tol)
    print('Combined output file path:', outpath)


def tsvcombine(tsvdirpath,filenames, outpath):
    '''
    our tsv files have no header
    '''
    tol=0
    # Open the output file in write mode
    with open(outpath, "w") as outfile:
        # Open and read each file line by line
        for filename in filenames:
            with open(tsvdirpath+filename+'.tsv', "r") as infile:
                infile=list(infile)
                flen=len(infile)
                tol+=flen
                fstartid=infile[0].split('\t')[0]
                fendid=infile[-1].split('\t')[0]
                print('>>'+filename+'.tsv<<'+',Length:',flen,', Start ID:', fstartid, ', End ID:', fendid)
                for line in infile:
                    outfile.write(line)
    print('Combined data length:', tol)
    print('Combined output file path:', outpath)

def plotscatter(fullcsvpath, outpath):
    '''
    plot the 2nd and 3rd columns as scatter plot and save the plot
    '''
    # Load the CSV file
    data = pd.read_csv(fullcsvpath)

    # Extract the second and third columns (assuming no header, otherwise use column names)
    # Add jitter to x and y values
    jitter_strength = 0.05  # Adjust this value as needed

    pred = data.iloc[:, 1]  
    round = data.iloc[:, 2] 
    pred = pred  + np.random.normal(0, jitter_strength, size=len(pred))# Second column
    round = round  + np.random.normal(0, jitter_strength, size=len(round))# Third column

    # Create a scatter plot
    plt.scatter(pred, round, alpha=0.1)

    # Label the axes
    plt.xlabel("Prediction(1 is True)")
    plt.ylabel("Round")
    plt.title("Scatter Plot of Prediction vs Round")

    # Save the plot as out.jpg
    plt.savefig(outpath)

    # Show the plot (optional, you can comment this out if you only want to save the file)
    plt.show()

def plotheat(fullcsvpath, outpath):
        
    data = pd.read_csv(fullcsvpath)

    # Create a pivot table that counts occurrences of each (Prediction, Round) pair
    heatmap_data = data.pivot_table(index='Round', columns='Prediction', aggfunc='size', fill_value=0)
    
    grand_total = heatmap_data.values.sum()

    # Calculate percentages based on the grand total
    heatmap_percentage = (heatmap_data / grand_total) * 100

    # Create annotation with counts and overall percentages
    annot = heatmap_data.astype(str) + "\n" + heatmap_percentage.round(1).astype(str) + '%'

    
    # Plot heatmap with discrete values
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data, annot=annot, fmt="", cmap="Blues", cbar_kws={'label': 'Count'})
    #sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues", cbar_kws={'label': 'Count'})
    plt.gca().invert_yaxis()
    # Set x and y axis labels
    plt.xlabel("Prediction (1 is True)")
    plt.ylabel("Round")
    plt.title("Heatmap of Prediction vs Round (Discrete)")

    # Save as 'out.jpg'
    plt.savefig(outpath)

    plt.show()



if __name__=='__main__':
    csvdirpath='/home/pwang71/pwang71/Koehn/table_qa/wikitq_eval_out/'
    tsvdirpath='/home/pwang71/pwang71/Koehn/table_qa/wikitq_out/'
    ##1-133, 134-219, 220-375, 376-419, 420-7341 (Base, no step by step)
    #filenames=['20241112001959','20241111143616','20241111022146', '20241111135531', '20241111023654']    
    
    ##Step by Step
    # 1-81, 82-206, 206-3315
    filenames=['20241112162008', '20241112180807', '20241112220201']
    ## Combine tsv files
    tsvoutpath=tsvdirpath+'StepbyStep_1-3315train_combined.tsv'
    csvoutpath=csvdirpath+'StepbyStep_1-3315train_combined.csv'
    #tsvcombine(tsvdirpath=tsvdirpath, filenames=filenames, outpath=tsvoutpath)
    #csvcombine(csvdirpath=csvdirpath, filenames=filenames, outpath=csvoutpath)
    # fullcombine(predpath='/home/pwang71/pwang71/Koehn/table_qa/wikitq_full_anal/StepbyStep_1-3315train_combined_pred.tsv', csvpath='/home/pwang71/pwang71/Koehn/table_qa/wikitq_eval_out/StepbyStep_1-3315train_combined.csv'
    #             ,outpath='/home/pwang71/pwang71/Koehn/table_qa/wikitq_full_anal/StepbyStep_1-3315train_combined_full.csv')
    #plotscatter(fullcsvpath='/home/pwang71/pwang71/Koehn/table_qa/wikitq_full_anal/1-7341train_combined_full.csv',
    #                 outpath='/home/pwang71/pwang71/Koehn/table_qa/wikitq_full_anal/1-7341train_PredVSRound_scatter.png')
    plotheat(fullcsvpath='/home/pwang71/pwang71/Koehn/table_qa/wikitq_full_anal/StepbyStep_1-3315train_combined_full.csv',
                  outpath='/home/pwang71/pwang71/Koehn/table_qa/wikitq_full_anal/StepbyStep_1-3315train_PredVSRound_heat.png')