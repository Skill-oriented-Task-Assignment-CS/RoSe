

"""

    WorkFlow                     : Model File------> skill decoding along with other columns-------------> data
    Dependency file for decoding : mapper file
    Tranning                     : Step 1 - create intermediate representation, collect mapper 
                                   Step 2 - train
                                   Step 3 - save the trained model
                                   .........................................................
                                   Step 4 - Load the saved model
                                   Step 5 - generate intermediate representation
                                   Step 6 - decode 


"""

import numpy as np
import pandas as pd
import math
import json
import pickle
from sdv.single_table import CTGANSynthesizer
from sdv.evaluation.single_table import evaluate_quality
from sdv.metadata import SingleTableMetadata




# index of upwork job dataset is 0
# index of upwork worker dataset is 1

mapper_files= []
model_files = ["lib/upwork_job.pkl" , "lib/upwork_worker.pkl"]
old_column_names_list = []
new_column_names_list = []
models= []


def save_list(l, fname) :
    with open(fname, "wb") as fp:   #Pickling
        pickle.dump(l, fp)

def load_list(fname) :
    with open(fname, "rb") as fp:   # Unpickling
        l = pickle.load(fp)
    return l


    
def load_clusters(filename):
    with open(filename, 'r') as file:
        loaded_dict = json.load(file)
        return loaded_dict

# load models
for model_file in model_files :
    model = CTGANSynthesizer.load(filepath=model_file)
    models.append(model)

# load mapper files
mapper_files.append(load_clusters("lib/upwork_jobs_mapper.json"))
mapper_files.append(load_clusters("lib/upwork_workers_mapper.json"))

# load column names
new_column_names_list.append(load_list("lib/new_column_names_upwork_jobs.dat"))
new_column_names_list.append(load_list("lib/new_column_names_upwork_workers.dat"))
old_column_names_list.append(load_list("lib/old_column_names_upwork_jobs.dat"))
old_column_names_list.append(load_list("lib/old_column_names_upwork_workers.dat"))



    



def get_skilllist(df):
    lines = list(df["Skill"])
    processed_lines = []
    for line in lines :
        tmp = [item.strip() for item in line.split(",") if len(item.strip())>=1]
        
        processed_lines.append(",".join(tmp))
    return processed_lines




def find_clusterID(cluster_skill_mapper, search_key):
    for key,value in cluster_skill_mapper.items():
        tmp = value[0]
        if search_key in tmp:
            return key
    return -1  # for val not found

def skillembedding_cluster(df,cluster_skill_mapper):
     """
       add extra columns(all unique clusterid) in each row , assign weigts 0 to inf
       for presence of the cluster elements   
       return the new df with per skill column and also delete the skills column
       and also return new column names
    """
     df_ = df.copy()

     new_columns = []
    
     #print(cluster_skill_mapper.keys())
     for cluster_id in list(cluster_skill_mapper.keys()):
        df_[cluster_id]=0
        new_columns.append(cluster_id)
    
     #print(df_.columns)   
     for index, row in df.iterrows():
        skills = row["Skill"].split(",")  # Split the skills by commas
        
        for skill in skills : 
            skill = skill.strip()
            #print(skill)
            cluster_id = find_clusterID(cluster_skill_mapper,skill)
            df_.at[index,cluster_id] = df_.at[index,cluster_id] + 1
     
     df_ = df_.drop('Skill', axis=1)
     return df_,new_columns

def GenerateData(dataset_id,rows):
    pass


def select_string_with_probability(strings, probabilities):
    selected_string = np.random.choice(strings, p=probabilities)
    return selected_string

def generte_strings(source_strings, probablities,n):
    len_ = len(source_strings)
    n = min(len_,n)
    
    generated_strings = set()
    for i in range(n):
        elected_String = select_string_with_probability(source_strings, probablities)
        while elected_String  in generated_strings :
            elected_String = select_string_with_probability(source_strings, probablities)
        generated_strings.add(elected_String)
    return list(generated_strings)
        

def skillDecoding_clustering(df, cluster_skill_mapper):
    """
      it takes intermediate representation of the skills of CTGAN and 
      return the df only for skills
    """
        
    arr = df.to_numpy()
    skil_column = []
    for i in range(arr.shape[0]):
        skills_arr = []
        
        for j in range(0,arr.shape[1]): 
            if arr[i][j]>=1:
                cluster_id = list(cluster_skill_mapper.keys())[j]
                probablities = cluster_skill_mapper[cluster_id][2]
                strings = cluster_skill_mapper[cluster_id][0]
                k = math.floor(arr[i][j])
                skills_arr.extend(generte_strings(strings, probablities,k))
        if len(skills_arr) == 0 :
            continue
        skills_arr = list(set(skills_arr))
        skills = ",".join(skills_arr)
        skil_column.append(skills)
        
    
    df = pd.DataFrame({'Skills': skil_column})
    return df

def join_output(skill_df,other_df):
    return pd.concat([skill_df, other_df], axis=1)

def generate(index,n):
    
    synthesizer = models[index]
    synthetic_data = synthesizer.sample(num_rows=int(n*1.5))
    generated_data = decode_back(synthetic_data, mapper_files[index], old_column_names_list[index], new_column_names_list[index])
    generated_data.dropna(inplace=True)
    return generated_data.sample(n)

    





def create_intermediate_representation(source_file_location, mapper_file_location ) :
    """
      input source file will dontains only the column which will be generated and the first column 
      should be the skills
    """
    df_source = pd.read_csv(source_file_location)
    df_source.dropna(inplace=True)

    print("************** Nan Stats : ", df_source.isnull().sum())

    cluster_mapper=load_clusters(mapper_file_location)



    df_intermediate, new_column_names = skillembedding_cluster(df_source,cluster_mapper) # this is ready for tranning

    column_names = df_intermediate.columns.tolist()

    old_column_names = list(set(column_names) - set(new_column_names))

    return df_intermediate, old_column_names,new_column_names



def decode_back(intermediate_df, mapper_fiel_location, old_column_names, new_Column_names) :
    skills_df = intermediate_df[new_Column_names]
    other_attrb_df = intermediate_df[old_column_names]
    skills = skillDecoding_clustering(skills_df,mapper_fiel_location)
    df = join_output(skills,other_attrb_df)
    return df

    


