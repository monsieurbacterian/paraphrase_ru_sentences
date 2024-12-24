from flask import Flask, request, render_template
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Загрузка модели и токенизатора
MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model.eval()  # Переводим модель в режим оценки

# Перефразирования текста
def paraphrase(text, beams=5, grams=4):
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams, num_beams=beams, max_length=max_size)
    return tokenizer.decode(out[0], skip_special_tokens=True)


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    original_text = ""  # Исходный текст
    paraphrased_text = ""  # Перефразированный текст

    if request.method == "POST":
        # Получаем текст из формы
        original_text = request.form.get("text", "")
        if original_text:
            # Перефразирование
            paraphrased_text = paraphrase(original_text)

    return render_template("main.html", original_text=original_text, paraphrased_text=paraphrased_text)#вывод

if __name__ == "__main__":
    app.run(debug=True)