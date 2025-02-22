from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
model_name = "facebook/bart-large-cnn"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
def summarize_text(text, max_length=150, min_length=50):
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
if __name__ == "__main__":
    print("Enter the article to summarize (Press Enter twice to end input):")
    input_text = ""
    while True:
        line = input()
        if line:
            input_text += line + " "
        else:
            break

    if input_text:
        print("\nOriginal Text:\n", input_text)
        summary = summarize_text(input_text)
        print("\nGenerated Summary:\n", summary)
    else:
        print("No text provided.")
