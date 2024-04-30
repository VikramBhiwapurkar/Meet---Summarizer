from datasets import load_dataset,load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from tqdm import tqdm
from datasets import load_dataset, load_metric
import pandas as pd
model = AutoModelForSeq2SeqLM.from_pretrained("C:\\ML\\Meeting Notes\\Pegasus-Model-20240408T200935Z-002\\Pegasus-Model", use_safetensors=True)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("C:\\ML\\Meeting Notes\\Tokenizer\\Tokenizer", tokenizer_config="C:\\ML\\Meeting Notes\\Tokenizer\\Tokenizer\\tokenizer_config.json")

# Create summarization pipeline
# summarization_pipeline = pipeline("summarization", model=model, tokenizer=tokenizer)

dataset_samsum = load_from_disk('samsum_dataset')

split_lengths = [len(dataset_samsum[split])for split in dataset_samsum]

print(f"Split lengths: {split_lengths}")
print(f"Features: {dataset_samsum['train'].column_names}")
print("\nDialogue:")

print(dataset_samsum["test"][1]["dialogue"])
print("\nSummary:")

print(dataset_samsum["test"][1]["summary"])

def convert_example_to_features(example_batch):
  input_encodings = tokenizer(example_batch['dialogue'], max_length = 10000, truncation=True)

  with tokenizer.as_target_tokenizer():
    target_encodings = tokenizer(example_batch['summary'],max_length = 200,truncation=True)

  return {
      'input_ids' : input_encodings['input_ids'],
      'attention_mask': input_encodings['attention_mask'],
      'labels': target_encodings['input_ids']
  }

dataset_samsum_pt = dataset_samsum.map(convert_example_to_features, batched = True)

def generate_batch_sized_chunks(list_of_elements,batch_size):
  """split the dataset into smaller batches that we can process simultaneously
  yield successive batch-sized chunks from list_of_elements."""
  for i in range(0,len(list_of_elements),batch_size):
    yield list_of_elements[i:i+batch_size]

def calculate_metric_on_test_ds(dataset,metric,model,tokenizer,
                                batch_size=16, device="cpu", column_text="article",
                                column_summary="highlights"):
  article_batches = list(generate_batch_sized_chunks(dataset[column_text],batch_size))
  target_batches = list(generate_batch_sized_chunks(dataset[column_summary],batch_size))

  for article_batch,target_batch in tqdm(
      zip(article_batches,target_batches), total=len(article_batches)):

      inputs = tokenizer(article_batch,max_length=1024,truncation = True,
                         padding="max_length",return_tensors="pt")

      summaries = model.generate(input_ids=inputs["input_ids"].to(device),
                        attention_mask=inputs["attention_mask"].to(device),
                        length_penalty = 0.8, num_beams=8, max_length=128)
      ''' parameter for length penalty ensures that the model does not generate sequences'''

      #Finally, we decode the generate texts,
      #replace the token, and add the decoded texts with the references to the metric
      decoded_summaries = [tokenizer.decode(s,skip_special_tokens=True,
                                   clean_up_tokenization_spaces=True)
                    for s in summaries]

      decoded_summaries = [d.replace(""," ") for d in decoded_summaries]

      metric.add_batch(predictions=decoded_summaries, references=target_batch)

      #Finally compute and return the ROUGE scores.
      score = metric.compute()
      return score

rouge_names = ["rouge1","rouge2","rougeL","rougeLsum"]
rouge_metric = load_metric('rouge')

score = calculate_metric_on_test_ds(dataset_samsum['test'][0:10], rouge_metric, model, tokenizer, batch_size = 2,column_text = 'dialogue',column_summary='summary')

rouge_dict = dict((rn,score[rn].mid.fmeasure) for rn in rouge_names)

pd.DataFrame(rouge_dict, index = [f'pegasus'])