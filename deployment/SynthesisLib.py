import numpy as np
import pandas as pd
import math
import json
import pickle
from sdv.single_table import CTGANSynthesizer

# Index of upwork job dataset is 0, index of upwork worker dataset is 1
mapper_files = []
model_files = ["lib/upwork_job.pkl", "lib/upwork_worker.pkl"]
old_column_names_list = []
new_column_names_list = []
models = []

# Helper function to save and load lists
def save_list(l, fname):
    with open(fname, "wb") as fp:
        pickle.dump(l, fp)

def load_list(fname):
    with open(fname, "rb") as fp:
        l = pickle.load(fp)
    return l

# Load clusters from JSON file
def load_clusters(filename):
    with open(filename, 'r') as file:
        loaded_dict = json.load(file)
        return loaded_dict

# Load models
for model_file in model_files:
    model = CTGANSynthesizer.load(filepath=model_file)
    models.append(model)

# Load mapper files and column names
mapper_files.append(load_clusters("lib/upwork_jobs_mapper.json"))
mapper_files.append(load_clusters("lib/upwork_workers_mapper.json"))
new_column_names_list.append(load_list("lib/new_column_names_upwork_jobs.dat"))
new_column_names_list.append(load_list("lib/new_column_names_upwork_workers.dat"))
old_column_names_list.append(load_list("lib/old_column_names_upwork_jobs.dat"))
old_column_names_list.append(load_list("lib/old_column_names_upwork_workers.dat"))

# Process skill list by cleaning and splitting skills
def get_skilllist(df):
    lines = list(df["Skill"])
    processed_lines = []
    for line in lines:
        tmp = [item.strip() for item in line.split(",") if len(item.strip()) >= 1]
        processed_lines.append(",".join(tmp))
    return processed_lines

# Find cluster ID for a given skill from the cluster-skill mapper
def find_clusterID(cluster_skill_mapper, search_key):
    for key, value in cluster_skill_mapper.items():
        tmp = value[0]
        if search_key in tmp:
            return key
    return -1  # Return -1 if not found

# Embed skills into clusters for further processing
def skillembedding_cluster(df, cluster_skill_mapper):
    df_ = df.copy()
    new_columns = []
    
    for cluster_id in list(cluster_skill_mapper.keys()):
        df_[cluster_id] = 0
        new_columns.append(cluster_id)
    
    for index, row in df.iterrows():
        skills = row["Skill"].split(",")  # Split the skills by commas
        for skill in skills:
            skill = skill.strip()
            cluster_id = find_clusterID(cluster_skill_mapper, skill)
            if cluster_id != -1:
                df_.at[index, cluster_id] += 1  # Safeguard: increment only if valid cluster_id
    
    df_ = df_.drop('Skill', axis=1)
    return df_, new_columns

# Helper function to generate strings based on probabilities
def select_string_with_probability(strings, probabilities):
    selected_string = np.random.choice(strings, p=probabilities)
    return selected_string

def generate_strings(source_strings, probabilities, n):
    len_ = len(source_strings)
    n = min(len_, n)
    
    generated_strings = set()
    for i in range(n):
        selected_string = select_string_with_probability(source_strings, probabilities)
        while selected_string in generated_strings:
            selected_string = select_string_with_probability(source_strings, probabilities)
        generated_strings.add(selected_string)
    return list(generated_strings)

# Decode skill clusters back into skills
def skillDecoding_clustering(df, cluster_skill_mapper):
    arr = df.to_numpy()
    skill_column = []
    for i in range(arr.shape[0]):
        skills_arr = []
        for j in range(0, arr.shape[1]):
            if arr[i][j] >= 1:
                cluster_id = list(cluster_skill_mapper.keys())[j]
                probabilities = cluster_skill_mapper[cluster_id][2]
                strings = cluster_skill_mapper[cluster_id][0]
                k = math.floor(arr[i][j])
                skills_arr.extend(generate_strings(strings, probabilities, k))
        if skills_arr:
            skills_arr = list(set(skills_arr))
            skills = ",".join(skills_arr)
            skill_column.append(skills)
    
    return pd.DataFrame({'Skills': skill_column})

# Join the skill DataFrame with other attributes
def join_output(skill_df, other_df):
    return pd.concat([skill_df, other_df], axis=1)

# Generate synthetic data based on the trained model
def generate(index, n):
    synthesizer = models[index]
    synthetic_data = synthesizer.sample(num_rows=int(n * 1.5))
    generated_data = decode_back(synthetic_data, mapper_files[index], old_column_names_list[index], new_column_names_list[index])
    generated_data.dropna(inplace=True)
    return generated_data.sample(n)

# Create intermediate representation for training
def create_intermediate_representation(source_file_location, mapper_file_location):
    df_source = pd.read_csv(source_file_location)
    df_source.dropna(inplace=True)

    print("************** Nan Stats : ", df_source.isnull().sum())
    cluster_mapper = load_clusters(mapper_file_location)

    df_intermediate, new_column_names = skillembedding_cluster(df_source, cluster_mapper)
    column_names = df_intermediate.columns.tolist()
    old_column_names = list(set(column_names) - set(new_column_names))

    return df_intermediate, old_column_names, new_column_names

# Decode intermediate representation back into original format
def decode_back(intermediate_df, mapper_file_location, old_column_names, new_column_names):
    skills_df = intermediate_df[new_column_names]
    other_attrb_df = intermediate_df[old_column_names]
    skills = skillDecoding_clustering(skills_df, mapper_file_location)
    df = join_output(skills, other_attrb_df)
    return df
