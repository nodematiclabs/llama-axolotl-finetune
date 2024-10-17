import os
from datasets import Dataset
from huggingface_hub import HfApi, create_repo, login

def read_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def create_dataset():
    data = {
        'input': [],
        'output': [],
        'instruction': []
    }
    
    for i in range(1, 7):
        story = read_file(f"data/story-{i}.txt")
        summary = read_file(f"data/summary-{i}.txt")
        
        data['input'].append(story)
        data['output'].append(summary)
        data['instruction'].append("Summarize the following story in my style")
    
    return Dataset.from_dict(data)

def publish_to_huggingface(dataset, repo_name):
    # Create a new repository on Hugging Face
    api = HfApi()
    create_repo(repo_name, repo_type="dataset", private=True)
    
    # Push the dataset to Hugging Face
    dataset.push_to_hub(repo_name)

# Main execution
if __name__ == "__main__":
    login('')
    dataset = create_dataset()
    repo_name = "nldemo/story-summarization-demo"  # Replace with your desired repository name
    publish_to_huggingface(dataset, repo_name)
    print(f"Dataset published to {repo_name} on Hugging Face")